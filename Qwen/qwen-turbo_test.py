# -*- coding: utf-8 -*-


from http import HTTPStatus
from openai import OpenAI

import os

import io
import sys
os.environ['NO_PROXY'] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
key_1 = "sk-a4dee6419dd34713828c7b362037b5cd"


def get_response_stu(instruction, prompt):
    client = OpenAI(
        api_key=key_1,  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope SDK的base_url
    )
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{'role': 'system', 'content': '你是一个关注中国中小学生学习情况的学习助手'},
                  {'role': 'user', 'content': '你是谁？'}]
    )
    return completion.choices[0].message.content


if __name__ == '__main__':
    input_to_qwen = input()
    print(input_to_qwen)


