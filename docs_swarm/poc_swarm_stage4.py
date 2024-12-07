import json
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional


client = OpenAI()

# Customer Service Routine
# --------------------------------------------stage 4: agent handoff----------------------------------------------------
# For the agent functions we've defined so far, like execute_refund or place_order they return a string, which will be provided to the model. 
# What if instead, we return an Agent object to indate which agent we want to transfer to

from utils.function_to_schema import function_to_schema

class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: str = "You are a helpful Agent"
    tools: list = []

def execute_refund(item_name):
    return "success"

def place_order(item_name):
    return "success"

# We can then update our code to check the return type of a function response, and if it's an Agent, update the agent in use! 
# Additionally, now run_full_turn will need to return the latest agent in use in case there are handoffs. 
# (We can do this in a Response class to keep things neat.)
class Response(BaseModel):
    agent: Optional[Agent]
    messages: list

class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: str = "You are a helpful Agent"
    tools: list = []

def execute_tool_call(tool_call, tools, agent_name):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"{agent_name}:", f"{name}({args})")

    return tools[name](**args)  # call corresponding function with provided arguments

refund_agent = Agent(
    name="Refund Agent",
    instructions="You are a refund agent. Help the user with refunds.",
    tools=[execute_refund],
)

sales_agent = Agent(
    name="Sales Assistant",
    instructions="You are a sales assistant. Sell the user a product.",
    tools=[place_order],
)

def transfer_to_refunds():
    return refund_agent

def run_full_turn(agent, messages):
    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()
    while True:
        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in current_agent.tools]
        tools = {tool.__name__: tool for tool in current_agent.tools}

        # === 1. get openai completion ===
        response = client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": current_agent.instructions}]
            + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)

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
                    f"Transfered to {current_agent.name}. Adopt persona immediately."
                )
            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)
    # ==== 3. return last agent used and new messages =====
    return Response(agent=current_agent, messages=messages[num_init_messages:])


agent = sales_agent
messages = []
while True:
    user = input("User: ")
    messages.append({"role": "user", "content": user})
    response = run_full_turn(agent, messages)
    agent = response.agent
    messages.extend(response.messages)
