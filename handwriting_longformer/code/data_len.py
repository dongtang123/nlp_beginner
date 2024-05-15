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
    print("mean length: ", int(math.ceil(mean)))
    print("median length", median)


if __name__ == "__main__":
    root_path = os.path.join('D:\\nlp_beginner\\handwriting_longformer\\data\\data_all')
    train_path = os.path.join(root_path, 'problem_all_depression_train.csv')
    dev_path = os.path.join(root_path, 'problem_all_depression_dev.csv')
    test_path = os.path.join(root_path, 'problem_all_depression_test.csv')
    max_mean_median(train_path)
    max_mean_median(dev_path)
    max_mean_median(test_path)
