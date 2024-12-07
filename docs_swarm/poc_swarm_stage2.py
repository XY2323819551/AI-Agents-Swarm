import json
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional

"""
core indea: a set of steps and the tools to execute them
"""

client = OpenAI()

# Customer Service Routine
# --------------------------------------------stage 2: add function calling----------------------------------------------------
# Models require functions to be formatted as a function schema. 
from utils.function_to_schema import function_to_schema

def look_up_item(search_query):
    """Use to find item ID.
    Search query can be a description or keywords."""
    return "item_132612938"

def execute_refund(item_id, reason="not provided"):
    print("Summary:", item_id, reason) # lazy summary
    return "success"

def execute_tool_call(tool_call, tools_map):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    print(f"Assistant: {name}({args})")
    return tools_map[name](**args)  # call corresponding function with provided arguments


system_message = (
    "You are a customer support agent for ACME Inc."
    "Always answer in a sentence or less."
    "Follow the following routine with the user:"
    "1. First, ask probing questions and understand the user's problem deeper.\n"
    " - unless the user has already provided a reason.\n"
    "2. Propose a fix (make one up).\n"
    "3. ONLY if not satesfied, offer a refund.\n"
    "4. If accepted, search for the ID and then execute refund."
    ""
)

tools = [execute_refund, look_up_item]
def run_full_turn(system_message, tools, messages):
    num_init_messages = len(messages)
    messages = messages.copy()
    while True:
        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in tools]
        tools_map = {tool.__name__: tool for tool in tools}
        # === 1. get openai completion ===
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_message}] + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)

        if message.content:  # print assistant response
            print("Assistant:", message.content)

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # === 2. handle tool calls ===
        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools_map)
            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)
    # ==== 3. return new messages =====
    return messages[num_init_messages:]

messages = []
while True:
    user = input("User: ")
    messages.append({"role": "user", "content": user})
    new_messages = run_full_turn(system_message, tools, messages)
    messages.extend(new_messages)
# Now that we have a routine, let's say we want to add more steps and more tools. 
# We can up to a point, but eventually if we try growing the routine with too many different tasks it may start to struggle. 
# This is where we can leverage the notion of multiple routines – given a user request, we can load the right routine with the appropriate steps and tools to address it.




# --------------------------------------------stage 3: a handoff as an agentn----------------------------------------------------



