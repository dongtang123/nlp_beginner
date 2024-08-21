from qwen_depression import instruction_and_prompt_depression
from qwen_anxiety import instruction_and_prompt_anxiety
from qwen_depression_judge import random_selection_phq, instruction_and_prompt_depression_judge, save_phq_score
from qwen_anxiety_judge import random_selection_gad, instruction_and_prompt_anxiety_judge, save_gad_score
from qwen_study import instruction_and_prompt_study_junior, instruction_and_prompt_study_senior, \
    instruction_and_prompt_study_to_psychology
from public_tool import get_response_openai, save_dialogue, read_dialogue, user_input
from user_simulator import instruction_and_prompt_user
import random


def random_start():  # user simulator 的一部分，用于开启对话
    greetings = [
        "你好，学习助手，我今天遇到了一些数学问题，能帮我解答吗？",
        "早上好，学习助手，我的英语作业有点难，能帮我解释一下吗？",
        "学习机器人，我在做数学作业时遇到了一些问题，可以指导我吗？",
        "你好，学习助手，能给我一些关于如何学习英语单词的建议吗？",
        "嗨，机器人，我需要帮助理解物理课上的一个概念，可以帮我吗？",
        "你好，我在准备数学考试，能不能给我一些复习的建议？",
        "学习助手，你好，我想知道如何更好地安排我的学习时间。",
        "你好，我在做作业时有些困惑，可以帮我解答一下吗？",
        "你好",
        "你好呀"
    ]
    selected_greeting = random.choice(greetings)
    print(selected_greeting)
    first_utterance = {"user": selected_greeting}
    save_dialogue(first_utterance)
    return


def user_flows():
    instruction, prompt = instruction_and_prompt_user()
    response_user = get_response_openai(instruction, prompt)
    response_user = {"user": response_user}
    print(response_user)
    save_dialogue(response_user)
    return


def user_flows_input():
    utterance = user_input()
    response_user = {"user": utterance}
    print(response_user)
    save_dialogue(response_user)


def study_flows(grade):
    if grade in [10, 11, 12]:
        instruction, prompt = instruction_and_prompt_study_senior()
    else:
        instruction, prompt = instruction_and_prompt_study_junior()
    print(instruction + prompt)
    response_study = get_response_openai(instruction, prompt)
    response_assistant = {"assistant": response_study}
    print(response_assistant)
    save_dialogue(response_assistant)
    return


def study_to_psychology_flows():
    instruction, prompt = instruction_and_prompt_study_to_psychology()
    print(instruction + prompt)
    response_study = get_response_openai(instruction, prompt)

    response_assistant = {"assistant": response_study}
    print(response_assistant)
    save_dialogue(response_assistant)
    return


def depression_flows():
    selected_index = random_selection_phq()
    instruction, prompt = instruction_and_prompt_depression(selected_index)
    print(instruction + prompt)
    response_depression = get_response_openai(instruction, prompt)
    response_assistant = {"assistant": response_depression}
    print(response_assistant)
    save_dialogue(response_assistant)
    return selected_index


def depression_judge(selected_index):
    dialogue_history = read_dialogue()
    instruction_judge, prompt_judge = instruction_and_prompt_depression_judge(selected_index, dialogue_history)

    response_depression_judge = get_response_openai(instruction_judge, prompt_judge)
    data = save_phq_score(selected_index, response_depression_judge)
    print(data)


def anxiety_flows():
    selected_index = random_selection_phq()
    instruction, prompt = instruction_and_prompt_anxiety(selected_index)
    print(instruction + prompt)
    response_depression = get_response_openai(instruction, prompt)
    response_assistant = {"assistant": response_depression}
    print(response_assistant)
    save_dialogue(response_assistant)
    return selected_index


def anxiety_judge(selected_index):
    dialogue_history = read_dialogue()
    instruction_judge, prompt_judge = instruction_and_prompt_anxiety_judge(selected_index, dialogue_history)

    response_depression_judge = get_response_openai(instruction_judge, prompt_judge)
    data = save_phq_score(selected_index, response_depression_judge)
    print(data)


def total_flow():
    for index in range(0, 10):

        if read_dialogue() is None:
            # random_start()

            user_flows_input()  # 用户输入
        if index < 4:
            print(f"study turn {index}:")
            study_flows(10)  # 传入年级，对应角色学习助手，分了junior和senior
            # user_flows()
            user_flows_input()
        elif index == 4:
            print(f"study to psychology turn {index}:")
            study_to_psychology_flows()  # 从学习转到关注心理状态
            user_flows_input()
        elif random_selection_phq() != "all_phq":
            print(f"depression turn {index}:")
            selected_index_depression = depression_flows()  # 针对抑郁提问，返回提问的问题的index
            # user_flows()
            user_flows_input()
            depression_judge(selected_index_depression)  # 针对回复进行抑郁判断
        elif random_selection_gad() != "all_gad":
            print(f"anxiety turn {index}:")
            selected_index_anxiety = anxiety_flows()  # 针对焦虑提问，返回提问的问题的index
            # user_flows()
            user_flows_input()
            anxiety_judge(selected_index_anxiety)  # 针对回复进行焦虑判断


if __name__ == "__main__":
    total_flow()  # 总流程
