import json
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
from llm_pool.llm import get_model_response_with_tools

# Customer Service Routine
from utils.function_to_schema import function_to_schema  #  使用正确的导入路径


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "deepseek-chat"  # deepseek-chat, mixtral-8x7b-32768, Qwen/Qwen2-72B-Instruct, gpt-4o, llama3-70b-8192
    instructions: str = "你是一个非常有用的人工智能助手，你使用中文回答用户的问题。"
    tools: list = []


class Response(BaseModel):
    agent: Optional[Agent]
    messages: list


def execute_order(product, price: int):
    """价格应该是人民币."""
    print("\n\n=== 订单详情 ===")
    print(f"商品: {product}")
    print(f"价格: ${price}")
    print("=================\n")
    confirm = input("确认订单? y/n: ").strip().lower()
    if confirm == "y":
        print("订单成功!")
        return "Success"
    else:
        print("订单取消!")
        return "User cancelled order."


def look_up_item(search_query):
    """用于查找商品ID。
    搜索查询可以是描述或关键字。"""
    item_id = "商品_132612938"
    print("找到商品:", item_id)
    return item_id


def execute_refund(item_id, reason="not provided"):
    print("\n\n=== 退款总结 ===")
    print(f"商品 ID: {item_id}")
    print(f"原因: {reason}")
    print("=================\n")
    print("退款成功!")
    return "success"


def escalate_to_human(summary):
    """只有在被明确要求时才会被调用。"""
    print("转接到人工服务......")
    print("\n=== 转接报告 ===")
    print(f"总结: {summary}")
    print("=========================\n")
    exit()


def transfer_back_to_triage():
    """如果客户提出了一个超出你权限范围的话题，包括转接人工服务的需求，就调用这个选项。"""
    return triage_agent


def transfer_to_sales_agent():
    """任何和销售、购买相关的客户"""
    return sales_agent


def transfer_to_issues_and_repairs_agent():
    """和问题，维修，或退款相关的客户。"""
    return issues_and_repairs_agent


triage_agent = Agent(
    name="Triage Agent",
    instructions=(
       "你是一个百货公司的客服机器人。你的自我介绍一定要非常简短，你的工作是收集信息，引导客户到正确的部门。你和客户交流的语气和善而自然。"
    ),
    tools=[transfer_to_sales_agent, transfer_to_issues_and_repairs_agent, escalate_to_human],
)


sales_agent = Agent(
    name="Sales Agent",
    instructions=(
        "你是百货公司的销售人员。始终用一句话或更短的语句回答。请按照以下流程与客户互动：1. 可以根据语境随时夸赞客户的性格、容貌等。2. 随意提及百货公司的一款热门产品可以帮助解决问题。 - 不要提及价格。3. 等客户对产品感兴趣后，抛出一个令人震惊的高价。4. 只有在所有步骤都完成并且客户同意购买后，才告知一个疯狂的附加条件并处理他们的订单。"),
    tools=[execute_order, transfer_back_to_triage],
)


issues_and_repairs_agent = Agent(
    name="Issues and Repairs Agent",
    instructions=(
        "你是百货公司的售后服务人员，负责解决售后问题，主要给客户提供维修方案或者退款。始终用一句话或更短的语句回答。请按照以下流程与客户互动：1. 首先提出探索性问题，深入了解客户的问题；- 除非客户已经说明了原因。2. 提出解决方案（可以创造性发挥）。3. 只有在客户对解决方案不满意时，才提供退款选项。4. 如果客户接受退款，查找订单ID然后执行退款。"
    ),
    tools=[execute_refund, look_up_item, transfer_back_to_triage],
)


def execute_tool_call(tool_call, tools, agent_name):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    print(f"{agent_name}:", f"{name}({args})")
    return tools[name](**args)  # call corresponding function with provided arguments


def run_full_turn(agent, messages):
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
        messages.append(message)  # 不仅仅是content字段

        if message.content:  # print agent response
            print(f"{current_agent.name}:", message.content)

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # === 2. handle tool calls ===
        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools, current_agent.name)
            if type(result) is Agent:  # if agent transfer, update current agent
                current_agent = result
                result = (
                    f"交接给 {current_agent.name}. 请立即进入角色."
                )
            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)
    # ==== 3. return last agent used and new messages =====
    # 这个函数直接对messages进行in-place操作也是可取的，都能争正确的返回message列表
    # 这里对message进行拆分，应该是为了更清晰地展示每一步的执行过程（response.messages是当前agent执行过程中产生的messages）
    return Response(agent=current_agent, messages=messages[num_init_messages:])


agent = triage_agent
messages = []

while True:
    user = input("客户: ")
    if user == "拜拜" or user == "q" or user == "exit":
        break
    messages.append({"role": "user", "content": user})
    response = run_full_turn(agent, messages)
    agent = response.agent
    messages.extend(response.messages)
