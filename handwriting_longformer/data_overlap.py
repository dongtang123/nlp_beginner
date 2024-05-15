import pandas as pd

import os


def overlap(path1, path2):
    df_1 = pd.read_csv(path1)
    df_2 = pd.read_csv(path2)
    for item1 in df_1['name']:
        for item2 in df_2['name']:
            if item1 == item2:
                print(item1)


if __name__ == "__main__":
    problem_1_depression_dev = os.path.join(
        '/data/caojieming/handwriting_long/handwriting_longformer/cross_validation_csv/problem_1_depression/dev_0.csv')
    problem_1_depression_test = os.path.join(
        '/data/caojieming/handwriting_long/handwriting_longformer/cross_validation_csv/problem_1_depression/test_0.csv')
    problem_1_depression_train = os.path.join(
        '/data/caojieming/handwriting_long/handwriting_longformer/cross_validation_csv/problem_1_depression/train_0.csv')
    overlap(problem_1_depression_dev, problem_1_depression_test)
    overlap(problem_1_depression_dev, problem_1_depression_train)
    overlap(problem_1_depression_test, problem_1_depression_train)
