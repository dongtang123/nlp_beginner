from qwen_anxiety import gad_scale
import json
import os
import random


def random_selection_gad():
    path = os.path.join("./temp_save/gad_score.json")
    if os.path.exists(path):
        with open(path, "r") as json_file:
            data = json.load(json_file)
    else:
        data = {"0": -1, "1": -1, "2": -1, "3": -1, "4": -1, "5": -1, "6": -1}
        with open(path, "w") as json_file:
            json.dump(data, json_file, indent=4)
    non_negative_items = [key for key, value in data.items() if value == -1 and value != "-1"]
    if non_negative_items:
        return int(random.choice(non_negative_items))
    else:
        return "all_gad"


def instruction_and_prompt_anxiety_judge(gad_problem_index, dialogue_history):
    gad_problem = gad_scale(gad_problem_index)
    instruction = f"""你是一个关注中国中小学生心理健康情况的学习心理医生，正在与中小学生进行对话。
请根据“对话历史”，，判断有关GAD-7中“相关问题”的情况。
你应该根据判断标准生成当前问题的评分。
你应该严格按照判断标准中的描述进行评分。
你的回答仅生成一个评分，而无需多余的任何信息。
"""
    prompt = f"""==========
对话历史如下：
{dialogue_history}

“相关问题”如下：
{gad_problem}

判断标准如下：
无法判断:-1
完全没有问题中的情况:0
过去两周有几天存在问题中的情况:1
过去两周超过一半时间存在问题中的情况:2
过去两周几乎每天存在问题中的情况:3
=========="""
    return instruction, prompt


def save_gad_score(index, response):
    path = os.path.join("./temp_save/gad_score.json")
    if os.path.exists(path):
        with open(path, "r") as json_file:
            data = json.load(json_file)
        index = str(index)
        data[index] = response
        with open(path, "w") as json_file:
            json.dump(data, json_file, indent=4)
    else:
        data = {"1": -1, "2": -1, "3": -1, "4": -1, "5": -1, "6": -1, "7": -1}
        with open(path, "w") as json_file:
            json.dump(data, json_file, indent=4)
    return data
