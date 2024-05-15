import pandas as pd

import os


def data_analysis(path):
    df = pd.read_csv(path)
    res_0, res_1, res_2 = 0, 0, 0
    for item in df['label']:
        if item == 0:
            res_0 += 1
        elif item == 1:
            res_1 += 1
        elif item == 2:
            res_2 += 1
    return res_0, res_1, res_2


if __name__ == "__main__":
    depression_path = os.path.join(
        'D:\\nlp_beginner\\handwriting_chatgpt\\aug_data\\data_1\\aug_problem_1_anxiety_train.csv')
    anxiety_path = os.path.join(
        'D:\\nlp_beginner\\handwriting_chatgpt\\aug_data\\data_1\\aug_problem_1_depression_train.csv')
    res_0, res_1, res_2 = data_analysis(depression_path)
    print('depression distribution is:')
    print(f"0: {res_0},1: {res_1},2: {res_2},")
    res_0, res_1, res_2 = data_analysis(anxiety_path)
    print('anxiety distribution is:')
    print(f"0: {res_0},1: {res_1},2: {res_2},")