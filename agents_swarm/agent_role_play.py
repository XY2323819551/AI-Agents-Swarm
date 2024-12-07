import json
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
from llm_pool.llm import get_model_response_with_tools
from datetime import datetime
import logging
from rich.console import Console
from rich.theme import Theme

# 禁用所有不需要的日志输出
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("arxiv").setLevel(logging.WARNING)
logging.getLogger("tool_pool.arxiv_pdf").setLevel(logging.WARNING)

# 创建自定义主题
custom_theme = Theme({
    "student": "cyan bold",
    "teacher": "green bold",
    "system": "yellow bold",
    "tool": "magenta bold",
})

console = Console(theme=custom_theme)

# 定义角色表情
ROLE_EMOJIS = {
    "Student Agent": "👨‍🎓",
    "Teacher Agent": "👨‍🏫",
    "System": "🤖",
    "Tool": "🛠️",
}

def print_role_message(role: str, message: str):
    """带颜色和表情符号打印角色消息"""
    emoji = ROLE_EMOJIS.get(role, "")
    style = role.lower().split()[0]  # 获取角色的基本名称作为样式
    console.print(f"{emoji} [{style}]{role}:[/] {message}")

def print_tool_call(agent_name: str, tool_name: str, args: dict):
    """打印工具调用信息"""
    emoji = ROLE_EMOJIS.get("Tool", "")
    console.print(f"{emoji} [{agent_name.lower().split()[0]}]{agent_name}[/] 调用工具: [tool]{tool_name}[/]({args})")

# Customer Service Routine
from utils.function_to_schema import function_to_schema  #  使用正确的导入路径
from utils.agent_logger import AgentLogger  # 在文件开头添加导入

# import tools
from tool_pool.arxiv import search_arxiv  # 根据关键词捞取论文metadata
from tool_pool.arxiv_pdf import get_arxiv_pdf_content  # 根据download—url获取内容


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"  # deepseek-chat, mixtral-8x7b-32768, Qwen/Qwen2-72B-Instruct, gpt-4o, llama3-70b-8192
    instructions: str = "你是一个非常有用的人工智能助手，你使用中文回答用户的问题。"
    tools: list = []


class Response(BaseModel):
    agent: Optional[Agent]
    messages: list


ROLE_PROMPTS = {
    "student": """你是一个热爱学习的AI研究生，对AI领域的前沿技术充满好奇。
行为准则：
1. 开场时，你需要先提出你的问题！
    可以从以下主题中选择一个发问：
   - Transformer架构及其变体
   - 大语言模型的训练技术
   - 提示词工程的最新进展
   - AI模型的对齐技术

2. 你喜欢追问，善于根据老师的回答，提出更深入的问题，这样可以全面的了解知。
3. 结束对话：
   - 当你觉得对某个主题理解透彻后
   - 用"感谢老师的讲解，我已经理解了"来结束对话
注意：每次只问一个焦点问题，避免同时问多个问题。""",



    "teacher": """你是一位专业的AI领域教授，擅长通过论文讲解技术概念。
角色特征：
- 深入理解AI领域的前沿技术
- 擅长结合论文解释复杂概念
- 耐心细致，循序渐进
- 善于启发学生思考

工具使用：
1. 当学生提出一个话题时，使用arxiv搜索工具查找相关论文
2. 如果学生继续追问，使用pdf内容提取工具，提取论文的具体内容，自己消化理解之后解答学生的问题
3. 如果你已经有了某篇论文的内容，不需要再次调用pdf内容提取工具，根据之前的论文内容回答学生的问题即可

回答准则：
1. 初次回答：
   - 使用arxiv工具搜索相关论文
   - 基于论文摘要提供概述性解释
   - 引用关键论文的结论

2. 当你解答完学生的问题之后，需要将话语权转接给学生

3. 深入解释：
   - 使用对应的工具获取获取论文详细内容
   - 结合论文具体章节进行解释
   - 提供实际应用案例

4. 互动规则：
   - 当学生表示理解时，鼓励继续探索
   - 当学生提出困惑时，提供更详细的解释
   - 当学生表示"已经理解了"时，回复"很高兴能帮助你理解这个概念"并结束对话
注意：回答要循序渐进，先解释基本概念，再深入技术细节。"""
}

SYSTEM_PROMPT = """这是一个AI教学场景的多轮对话系统。对话规则：

1. 对话流程：
   - 学生首先提出一个AI技术相关的问题，比如：我想了解一下Transformer架构
   - 教师基于arxiv论文回答问题
   - 学生可以继续追问或表示理解
   - 当学生表示理解后对话结束

2. 角色转换：
   - 每轮对话明确标识说话角色
   - 严格遵循学生-教师交替发言

3. 对话限制：
   - 每轮对话聚焦于单一主题
   - 每次回复控制在300字以内
   - 最多进行10轮问答
   - 使用中文

4. 结束条件：
   - 学生明确表示理解
   - 教师确认回复"很高兴能帮助你理解这个概念"
"""


def transfer_to_student_agent():
    """轮到学生student发言了"""
    return student_agent


def transfer_to_teacher_agent():
    """轮到教师teacher发言了"""
    return teacher_agent


teacher_agent = Agent(
    name="Teacher Agent",
    instructions=ROLE_PROMPTS["teacher"],
    tools=[transfer_to_student_agent, search_arxiv, get_arxiv_pdf_content]
)


student_agent = Agent(
    name="Student Agent",
    instructions=ROLE_PROMPTS["student"],
    tools=[transfer_to_teacher_agent],
)


def execute_tool_call(tool_call, tools, agent_name):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    print_tool_call(agent_name, name, args)
    return tools[name](**args)


def run_full_turn(agent, messages, logger):
    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()
    
    while True:
        tool_schemas = [function_to_schema(tool) for tool in current_agent.tools]
        tools = {tool.__name__: tool for tool in current_agent.tools}
        
        response = get_model_response_with_tools(
            model_name=current_agent.model,
            messages=[{"role": "system", "content": current_agent.instructions}]
            + messages,
            tools=tool_schemas or None
        )
        message = response.choices[0].message
        messages.append(message)

        logger.start_agent_session(current_agent.name)

        if message.content:
            print_role_message(current_agent.name, message.content)
            logger.log_agent_message(current_agent.name, message.content)

        if not message.tool_calls:
            break

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools, current_agent.name)
            
            logger.log_tool_call(tool_call.function.name, 
                               json.loads(tool_call.function.arguments), 
                               result)
            
            if type(result) is Agent:
                current_agent = result
                result = f"交接给 {current_agent.name}. 请立即进入角色."
            
            if not isinstance(result, str):
                result = str(result)
                
            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)
    
    return Response(agent=current_agent, messages=messages[num_init_messages:])


def run_conversation():
    """运行自主对话"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    logger = AgentLogger()
    max_rounds = 20
    current_round = 0
    current_agent = student_agent
    
    while current_round < max_rounds:
        logger.start_new_round()
        console.print(f"\n[bold yellow]=== 对话轮次 {current_round + 1} ===[/]")
        
        response = run_full_turn(current_agent, messages, logger)
        
        last_message_content = ""
        if response.messages:
            last_message = response.messages[-1]
            if isinstance(last_message, dict):
                last_message_content = last_message.get("content", "")
            else:
                last_message_content = last_message.content if hasattr(last_message, 'content') else ""
        
        if "感谢老师的讲解，我已经理解了" in last_message_content:
            console.print("\n[cyan bold]学生表示已理解，对话结束[/]")
            log_file = logger.save_log()
            break
        if "很高兴能帮助你理解这个概念" in last_message_content:
            console.print("\n[green bold]教师确认结束，对话完成[/]")
            log_file = logger.save_log()
            break
            
        messages.extend(response.messages)
        current_agent = response.agent
        current_round += 1
    
    log_file = logger.save_log()
    console.print(f"\n[yellow]对话日志已保存到: {log_file}[/]")

if __name__ == "__main__":
    console.print("[bold yellow]开始AI教学对话...[/]")
    console.print("=" * 50)
    run_conversation()
