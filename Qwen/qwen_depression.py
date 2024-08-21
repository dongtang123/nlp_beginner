from public_tool import read_dialogue


def phq_scale(index):
    list_phq = ["1.做什么事都没兴趣, 沒意思", "2.感到心情低落, 抑郁, 沒希望", "3.入睡困难,总是醒着, 或睡得太多嗜睡",
                "4.常感到很疲倦,沒劲",
                "5.口味不好,或吃的太多", "6.自己对自己不满, 觉得自己是个失败者,或让家人丟脸了",
                "7.无法集中精力,即便是读报纸或看电视时,记忆力下降",
                "8.行动或说话缓慢到引起人们的注意,或刚好相反, 坐臥不安,烦躁易怒易怒,到处走动",
                "9.有不如一死了之的念头, 或想怎样伤害自己一下"]
    return list_phq[index]


def instruction_and_prompt_depression(phq_problem_index):
    dialogue_history = read_dialogue()
    phq_problem = phq_scale(phq_problem_index)
    instruction = f"""你是一位心理专家，你正在与一位中学生对话，同时也是一位中学老师。
你现在需要询问PHQ-9量表中的问题：{phq_problem}。
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


def speak_skills_depression(index):
    skills_list = [{"0": "委婉询问"}, {"1": "委婉询问"}, {"2": "委婉询问"}, {"3": "委婉询问"}, {"4": "委婉询问"},
                   {"5": "委婉询问"}, {"6": "委婉询问"}, {"7": "委婉询问"}, {"8": "委婉询问"}]
    return skills_list[index][str(index)]
