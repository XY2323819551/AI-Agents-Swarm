import json
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
from llm_pool.llm import get_model_response_with_tools
from datetime import datetime

# Customer Service Routine
from utils.function_to_schema import function_to_schema  #  使用正确的导入路径

# 在文件开头添加导入
from utils.agent_logger import AgentLogger


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"  # deepseek-chat, mixtral-8x7b-32768, Qwen/Qwen2-72B-Instruct, gpt-4o, llama3-70b-8192
    instructions: str = "你是一个非常有用的人工智能助手，你使用中文回答用户的问题。"
    tools: list = []


class Response(BaseModel):
    agent: Optional[Agent]
    messages: list


# 工具1: 查询天气
from tool_pool.weather import get_weather


def get_data_info(location: str) -> str:
    """获取当前日期信息
    
    Args:
        location (str): 地点参数(当前函数未使用此参数，保留是为了接口一致性)
        
    Returns:
        str: 返回格式化的日期字符串，格式为"YYYY年MM月DD日"
        
    Example:
        >>> get_data_info("北京")
        '2024年03月21日'
    """
    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y年%m月%d日")
    return formatted_date

# 工具2: 诗歌创作
def poetry_creation(weather: str, location: str, date: str) -> str:
    """根据天气、地点和当前的时间信息创作一首诗。
    
    Args:
        weather (str): 天气状况
        location (str): 地点名称
        date (str): 当前日期

    Returns:
        str: 创作的诗歌内容
    """
    # 如果没有提供日期，使用当前日期
    if not date:
        date = datetime.now().strftime("%Y年%m月%d日")
    
    # 确保参数不为空字符串
    weather = weather or "晴天"
    location = location or "塞纳河畔"
    poem = get_model_response_with_tools(
        model_name="deepseek-chat",
        messages=[
            {"role": "system", "content": """你是一位才华横溢的诗人，擅长捕捉当下的时空之美。创作时请注意：
            1. 准确把握季节特征，将天气、温度等自然元素融入诗中
            2. 体现地域特色，紧密结合当前地点，展现当地独特的景观和人文气息
            3. 用优美的意象和细腻的感受打动读者
            4. 根据用户需求灵活选择诗歌形式（古诗、现代诗等）"""}
        ] + [{"role": "user", "content": f"请根据以下信息创作一首诗：\n地点：{location}\n天气状况：{weather}，当前日期：{date}"}]
    )
    return poem


def transfer_back_to_triage():
    """如果用户提出了一个超出你职责的话题，就调用这个选项。"""
    return triage_agent


def transfer_to_poem_agent():
    """交接给诗歌创作agent"""
    return poem_agent


def transfer_to_get_env_agent():
    """交接给环境信息查询agent"""
    return get_env_agent


triage_agent = Agent(
    name="Triage Agent",
    instructions=(
       "你是一个专门分发任务的小助手，擅长使用工具。你的工作是收集信息，调用不同的agent来解决用户的问题。你和用户交流的语气和善而自然。比如用户说：我在上海，请根据今天的天气信息等给我写一首现代诗，你可以说：好的，我需要先查询一下今天的上海的环境信息……等得到天气信息之后，再使用诗歌创作agent进行诗歌创作(优先调用工具)"
    ),
    tools=[transfer_to_get_env_agent, transfer_to_poem_agent],
)


get_env_agent = Agent(
    name="Get Weather Agent",
    instructions=("你的任务根据用户所在地点，收集该地的环境信息，你擅长使用工具。你需要 \
                  1. 根据用户所在地点进行天气的查询，并返回天气信息 \
                  2. 获取当前日期信息 \
                  3. 交接给诗歌创作agent \
                  优先调用工具。"),
    tools=[get_weather, get_data_info, transfer_to_poem_agent, transfer_back_to_triage]
)


poem_agent = Agent(
    name="Poetry Creation Agent",
    instructions=(
        "你的职责是根据天气信息和地点信息写一首诗。如果用户没有指明诗歌的风格，就写一首现代诗；如果用户有要求诗歌的风格，则按照用户要求的风格写诗。\
        你不会回答你职责以外的问题，但是会调用合适的工具进行任务交接。"
    ),
    tools=[poetry_creation, transfer_back_to_triage],
)


def execute_tool_call(tool_call, tools, agent_name):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    print(f"{agent_name}:", f"{name}({args})")
    return tools[name](**args)  # call corresponding function with provided arguments


def run_full_turn(agent, messages, logger):
    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()
    
    while True:
        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in current_agent.tools]
        tools = {tool.__name__: tool for tool in current_agent.tools}
        
        # === 1. get openai completion ===
        response = get_model_response_with_tools(
            model_name=current_agent.model,
            messages=[{"role": "system", "content": current_agent.instructions}]
            + messages,
            tools=tool_schemas or None
        )
        message = response.choices[0].message
        messages.append(message)

        # 在收到LLM响应后，确保当前agent会话已经开始
        logger.start_agent_session(current_agent.name)

        if message.content:  # print agent response
            print(f"{current_agent.name}:", message.content)
            # 记录agent的回复
            logger.log_agent_message(current_agent.name, message.content)

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # === 2. handle tool calls ===
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            result = execute_tool_call(tool_call, tools, current_agent.name)
            
            # 记录工具调用的结果
            logger.log_tool_call(tool_name, tool_args, result)
            
            if type(result) is Agent:
                current_agent = result
                result = f"交接给 {current_agent.name}. 请立即进入角色."
            
            # 确保result是字符串类型
            if not isinstance(result, str):
                result = str(result)
                
            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)
    
    return Response(agent=current_agent, messages=messages[num_init_messages:])


agent = triage_agent
messages = []

# 创建logger实例
logger = AgentLogger()

print("我是一个可以根据地点写诗的小助手，你可以说：我在上海，请给我写一首七言绝句")
while True:
    user = input("user: ")
    if user in ["ok", "q", "exit", "好的", "好的谢谢"]:
        # 在对话结束时保存日志
        log_file = logger.save_log()
        print(f"日志已保存到: {log_file}")
        break
    
    # 在处理新的用户输入前，创建新的轮次（第一轮除外）
    if len(messages) > 0:  # 如果不是第一轮对话
        logger.start_new_round()
        
    # 记录用户输入
    logger.log_user_message(user)
    messages.append({"role": "user", "content": user})
    
    # 执行agent对话
    response = run_full_turn(agent, messages, logger)
    agent = response.agent
    messages.extend(response.messages)
