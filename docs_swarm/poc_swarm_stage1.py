import json
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional

"""
core idea: a set of steps and the tools to execute them
"""

client = OpenAI()

# Customer Service Routine
# --------------------------------------------stage 1----------------------------------------------------
# 例程的主要优势在于其简单性和鲁棒性。
# 注意，这些指令包含条件判断，很像状态机或代码中的分支。
# LLM实际上可以很好地处理小型和中型例程中的这些情况，还有一个额外的好处是具有"软"遵从性 - LLM可以自然地引导对话而不会陷入死胡同。
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

# To execute a routine, let's implement a simple loop that goes through the steps.
# 1. Gets user input. 获取用户输入。
# 2. Appends user message to messages.
# 3. Calls the model. 调用模型。
# 4. Appends model response to messages.

def run_full_turn(system_message, messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message}
        ] + messages,
    )
    message = response.choices[0].message
    messages.append(message)

    if message.content: print("Assistant:", message.content)
    return message


messages = []
while True:
    user = input("User: ")
    messages.append({"role": "user", "content": user})
    run_full_turn(system_message, messages)
