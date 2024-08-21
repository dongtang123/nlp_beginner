from public_tool import read_dialogue


def gad_scale(index):
    list_gad = ["1.感觉紧张，焦虑或急切", "2.不能停止或无法控制担心", "3.对很多不同的事情担忧", "4.很紧张，很难放松下来",
                "5.非常焦躁，以至无法静坐", "6.变得容易烦恼或易被激怒", "7.感到好像有什么可怕的事会发生"]
    return list_gad[index]


def instruction_and_prompt_anxiety(gad_problem_index):
    dialogue_history = read_dialogue()
    gad_problem = gad_scale(gad_problem_index)
    instruction = f"""你是一位心理专家，你正在与一位中学生对话，同时也是一位中学老师。
你现在需要问GAD-9量表中的问题：{gad_problem}。
你不能直白地询问，你应该根据对话历史进行询问。
你应该最关注对话历史中对方的最后一句，并保证提问平滑、委婉且不突兀。
你应该保证你的回复通俗易懂，因为对方是中学生。
你应该保持回复简短。
"""
    prompt = f"""==========
对话历史：
{dialogue_history}
=========="""
    return instruction, prompt


def speak_skills_anxiety(index):
    skills_list = [{"0": "委婉询问"}, {"1": "委婉询问"}, {"2": "委婉询问"}, {"3": "委婉询问"}, {"4": "委婉询问"},
                   {"5": "委婉询问"}, {"6": "委婉询问"}]
    return skills_list[index][str(index)]
