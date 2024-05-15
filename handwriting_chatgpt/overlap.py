import pandas as pd
import os


def overlap(path_left, path_right):
    df_left = pd.read_csv(path_left)
    df_right = pd.read_csv(path_right)

    # 执行左连接操作，保留df_left中name包含但df_right中user_id不包含的元素
    result = pd.merge(df_left, df_right, left_on='name', right_on='name', how='left', indicator=True)
    result = result[result['_merge'] == 'left_only']

    # 提取结果中的name列
    desired_elements_left = result['name'].tolist()
    # 执行左连接操作，保留df_right中name包含但df_left中user_id不包含的元素
    result_right = pd.merge(df_right, df_left, left_on='name', right_on='name', how='left', indicator=True)
    result_right = result_right[result_right['_merge'] == 'left_only']

    # 提取结果中的name列
    desired_elements_right = result_right['name'].tolist()

    # 交集
    list_left = [item for item in df_left['name']]
    list_right = [item for item in df_right['name']]
    intersection = set(list_left) & set(list_right)
    return desired_elements_left, desired_elements_right, intersection


if __name__ == "__main__":
    problem_1_anxiety_path = os.path.join(
        'D:\\nlp_beginner\\handwriting_chatgpt\\data\\data_1\\problem_1_anxiety_train.csv')
    problem_1_depression_path = os.path.join(
        'D:\\nlp_beginner\\handwriting_chatgpt\\data\\data_1\\problem_1_depression_train.csv')
    left, right, intersection = overlap(problem_1_anxiety_path,problem_1_depression_path)
    print(left)
    print(len(left))
    print(right)
    print(intersection)
