import pandas as pd
import os
import math


def max_mean_median(path):
    df = pd.read_csv(path)
    list_length = [len(item) for item in df['text']]
    max_value = max(list_length)
    mean = sum(list_length) / len(list_length)
    sorted_data = sorted(list_length)
    n = len(sorted_data)
    if n % 2 == 0:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        median = sorted_data[n // 2]
    print("max length: ", max_value)
    print("mean length: ",int(math.ceil(mean)))
    print("median length", median)


if __name__ == "__main__":
    # print("problem_01")
    # problem_1_path = os.path.join('data/problem_01.csv')
    # max_mean_median(problem_1_path)
    # print("problem_02")
    # problem_2_path = os.path.join('data/problem_02.csv')
    # max_mean_median(problem_2_path)
    # print("problem_03")
    # problem_3_path = os.path.join('data/problem_03.csv')
    # max_mean_median(problem_3_path)
    # print("problem_04")
    # problem_4_path = os.path.join('data/problem_04.csv')
    # max_mean_median(problem_4_path)
    # print("problem_05")
    # problem_5_path = os.path.join('data/problem_05.csv')
    # max_mean_median(problem_5_path)
    # print("problem_past_future")
    # problem_past_future_path = os.path.join('data/problem_past_future.csv')
    # max_mean_median(problem_past_future_path)
    problem_all = os.path.join('data/binary_aug_problem_1_2_depression_train.csv')
    max_mean_median(problem_all)

