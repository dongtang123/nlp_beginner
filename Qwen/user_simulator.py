import json
import os
from public_tool import read_dialogue


def instruction_and_prompt_user():
    dialogue_history = read_dialogue()
    instruction = "你是一名初二的初中生，正在与一个聊天机器人对话。你应该根据自身情况和对话历史进行回答或者提问。你的回复都应该保持简短，你应该最关注对话历史中对方的最后一句。"
    prompt = f"""==========
对话历史：{dialogue_history}
自身情况：有一点抑郁，但是并不焦虑。
=========="""
    return instruction, prompt


