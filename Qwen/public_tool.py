from http import HTTPStatus
from openai import OpenAI
import dashscope
import os
import datetime
import io
import sys
import json
import os

os.environ['NO_PROXY'] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
key_1 = "sk-a4dee6419dd34713828c7b362037b5cd"
dashscope.api_key = key_1


def save_dialogue(response_dict):
    path = os.path.join("./temp_save/dialogue.json")
    if os.path.exists(path):
        with open(path, "r", encoding='utf8') as json_file:
            data = json.load(json_file)
        data.append(response_dict)
        with open(path, "w", encoding='utf8') as json_file:
            json.dump(data, json_file, ensure_ascii=False,  indent=4)
    else:
        data = [response_dict]
        with open(path, "w", encoding='utf8') as json_file:
            json.dump(data, json_file,  ensure_ascii=False, indent=4)


def read_dialogue():
    path = os.path.join("./temp_save/dialogue.json")
    if os.path.exists(path):
        with open(path, "r", encoding='utf8') as json_file:
            data = json.load(json_file)
        return data
    else:
        return None


def get_response_openai(instruction, prompt):
    start_time = datetime.datetime.now()
    client = OpenAI(
        api_key=key_1,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{'role': 'system', 'content': instruction},
                  {'role': 'user', 'content': prompt}],
        temperature=0,
        top_p=0.8,
        seed=0
    )
    end_time = datetime.datetime.now()
    utterance = completion.choices[0].message.content
    print(end_time - start_time)
    return utterance



def user_input():
    input_to_qwen = input()
    return input_to_qwen


def get_response_dashscope(instruction, prompt):
    utterance = ""
    start_time = datetime.datetime.now()
    responses = dashscope.Generation.call("qwen-plus",
                                          messages=[{'role': 'system', 'content': instruction},
                                                    {'role': 'user', 'content': prompt}],
                                          result_format='message',  # set the result to be "message"  format.
                                          stream=True,  # set streaming output
                                          incremental_output=True,  # get streaming output incrementally
                                          temperature=0,
                                          top_p=0.8,
                                          seed=0
                                          )
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            end_time = datetime.datetime.now()
            print(end_time - start_time)
            print(response.output.choices[0]['message']['content'], end='')
            utterance += response.output.choices[0]['message']['content']
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
    return {"assistant": utterance}
