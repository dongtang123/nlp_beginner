# -*- coding: utf-8 -*-


from http import HTTPStatus
from openai import OpenAI
import dashscope
import os
import datetime
import io
import sys
from public_tool import get_response_dashscope, get_response_openai,read_dialogue

os.environ['NO_PROXY'] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
key_1 = "sk-a4dee6419dd34713828c7b362037b5cd"
dashscope.api_key = key_1



def instruction_and_prompt_study_to_psychology():
    dialogue_history = read_dialogue()
    instruction = """你是心理医生，正在与学生进行对话。
前文你们聊了学习情况，现在请根据对话历史，将话题转移到关心学生的心理状态上来。
你的话题转移应该平滑且委婉，不能直接询问对方心理状态如何。
"""
    prompt = f"""==========
对话历史：{dialogue_history}
=========="""
    return instruction, prompt

def instruction_and_prompt_study_junior():
    dialogue_history = read_dialogue()
    instruction = """你是一个关注中国初中学生学习情况的学习助手，正在与初中学生进行对话。
请根据对话历史，询问他关于他的学习情况。
若对方提问应该对初中生现有的疑问做出解答。
你应该更关注对话历史中user的最后一句回复。
保证回复简短，且通俗易懂，因为对方是中学生。
"""
    prompt = f"""==========
对话历史：{dialogue_history}
=========="""
    return instruction, prompt


def instruction_and_prompt_study_senior():
    dialogue_history = read_dialogue()
    instruction = """你是一个关注中国高中学生学习情况的学习助手，正在与高中学生进行对话。
请根据对话历史，询问他关于他的学习情况。
若对方提问应该对高中生现有的疑问做出解答。
你应该更关注对话历史中user的最后一句回复。
"""
    prompt = f"""==========
对话历史：{dialogue_history}
=========="""
    return instruction, prompt


