import pandas as pd
import os
from sklearn.model_selection import train_test_split
import random


def triple_classification(dataframe, column, train_path, dev_path, test_path, test_len=12, dev_len=8):
    name = [item for item in dataframe['name']]
    text = [item for item in dataframe['content']]
    label = [item for item in dataframe[column]]
    train_name, train_text, train_label = [], [], []
    dev_name, dev_text, dev_label = [], [], []
    test_name, test_text, test_label = [], [], []
    count_dev_0, count_test_0 = 0, 0
    count_dev_1, count_test_1 = 0, 0
    count_dev_2, count_test_2 = 0, 0
    for item1, item2, item3 in zip(name, text, label):
        if item3 == 2 and count_test_2 < test_len:
            test_label.append(item3)
            test_text.append(item2)
            test_name.append(item1)
            count_test_2 += 1
        elif item3 == 2 and count_dev_2 < dev_len:
            dev_label.append(item3)
            dev_text.append(item2)
            dev_name.append(item1)
            count_dev_2 += 1
        elif item3 == 2:
            train_label.append(item3)
            train_text.append(item2)
            train_name.append(item1)

        elif item3 == 1 and count_test_1 < test_len:
            test_label.append(item3)
            test_text.append(item2)
            test_name.append(item1)
            count_test_1 += 1
        elif item3 == 1 and count_dev_1 < dev_len:
            dev_label.append(item3)
            dev_text.append(item2)
            dev_name.append(item1)
            count_dev_1 += 1
        elif item3 == 1:
            train_label.append(item3)
            train_text.append(item2)
            train_name.append(item1)

        elif item3 == 0 and count_test_0 < test_len:
            test_label.append(item3)
            test_text.append(item2)
            test_name.append(item1)
            count_test_0 += 1
        elif item3 == 0 and count_dev_0 < dev_len:
            dev_label.append(item3)
            dev_text.append(item2)
            dev_name.append(item1)
            count_dev_0 += 1
        elif item3 == 0:
            train_label.append(item3)
            train_text.append(item2)
            train_name.append(item1)
        train_df = {'name': train_name, 'text': train_text, 'label': train_label}
        train_df = pd.DataFrame(train_df)
        dev_df = {'name': dev_name, 'text': dev_text, 'label': dev_label}
        dev_df = pd.DataFrame(dev_df)
        test_df = {'name': test_name, 'text': test_text, 'label': test_label}
        test_df = pd.DataFrame(test_df)
        train_df.to_csv(train_path, index=False)
        dev_df.to_csv(dev_path, index=False)
        test_df.to_csv(test_path, index=False)


def binary_classification(dataframe, column, train_path, dev_path, test_path, test_len=12, dev_len=8):
    name = [item for item in dataframe['name']]
    text = [item for item in dataframe['content']]
    label = [item for item in dataframe[column]]
    train_name, train_text, train_label = [], [], []
    dev_name, dev_text, dev_label = [], [], []
    test_name, test_text, test_label = [], [], []
    count_dev_0, count_test_0 = 0, 0
    count_dev_1, count_test_1 = 0, 0
    for item1, item2, item3 in zip(name, text, label):
        if item3 == 1 and count_test_1 < test_len:
            test_label.append(item3)
            test_text.append(item2)
            test_name.append(item1)
            count_test_1 += 1
        elif item3 == 1 and count_dev_1 < dev_len:
            dev_label.append(item3)
            dev_text.append(item2)
            dev_name.append(item1)
            count_dev_1 += 1
        elif item3 == 1:
            train_label.append(item3)
            train_text.append(item2)
            train_name.append(item1)

        elif item3 == 0 and count_test_0 < test_len:
            test_label.append(item3)
            test_text.append(item2)
            test_name.append(item1)
            count_test_0 += 1
        elif item3 == 0 and count_dev_0 < dev_len:
            dev_label.append(item3)
            dev_text.append(item2)
            dev_name.append(item1)
            count_dev_0 += 1
        elif item3 == 0:
            train_label.append(item3)
            train_text.append(item2)
            train_name.append(item1)

        train_df = {'name': train_name, 'text': train_text, 'label': train_label}
        train_df = pd.DataFrame(train_df)
        dev_df = {'name': dev_name, 'text': dev_text, 'label': dev_label}
        dev_df = pd.DataFrame(dev_df)
        test_df = {'name': test_name, 'text': test_text, 'label': test_label}
        test_df = pd.DataFrame(test_df)
        train_df.to_csv(train_path, index=False)
        dev_df.to_csv(dev_path, index=False)
        test_df.to_csv(test_path, index=False)


def life_satisfactory(dataframe, train_path, dev_path, test_path, test_len=3, dev_len=3):
    name = [item for item in dataframe['name']]
    text = [item for item in dataframe['content']]
    label = [item for item in dataframe['生活满意度']]
    train_name, train_text, train_label = [], [], []
    dev_name, dev_text, dev_label = [], [], []
    test_name, test_text, test_label = [], [], []
    count_dev_0, count_test_0 = 0, 0
    count_dev_1, count_test_1 = 0, 0
    count_dev_2, count_test_2 = 0, 0
    count_dev_3, count_test_3 = 0, 0
    count_dev_4, count_test_4 = 0, 0
    count_dev_5, count_test_5 = 0, 0
    count_dev_6, count_test_6 = 0, 0
    for item1, item2, item3 in zip(name, text, label):
        if item3 == 6 and count_test_6 < test_len:
            test_label.append(item3)
            test_text.append(item2)
            test_name.append(item1)
            count_test_6 += 1
        elif item3 == 6 and count_dev_6 < dev_len:
            dev_label.append(item3)
            dev_text.append(item2)
            dev_name.append(item1)
            count_dev_6 += 1
        elif item3 == 6:
            train_label.append(item3)
            train_text.append(item2)
            train_name.append(item1)
        elif item3 == 5 and count_test_5 < test_len:
            test_label.append(item3)
            test_text.append(item2)
            test_name.append(item1)
            count_test_5 += 1
        elif item3 == 5 and count_dev_5 < dev_len:
            dev_label.append(item3)
            dev_text.append(item2)
            dev_name.append(item1)
            count_dev_5 += 1
        elif item3 == 5:
            train_label.append(item3)
            train_text.append(item2)
            train_name.append(item1)

        elif item3 == 4 and count_test_4 < test_len:
            test_label.append(item3)
            test_text.append(item2)
            test_name.append(item1)
            count_test_4 += 1
        elif item3 == 4 and count_dev_4 < dev_len:
            dev_label.append(item3)
            dev_text.append(item2)
            dev_name.append(item1)
            count_dev_4 += 1
        elif item3 == 4:
            train_label.append(item3)
            train_text.append(item2)
            train_name.append(item1)

        elif item3 == 3 and count_test_3 < test_len:
            test_label.append(item3)
            test_text.append(item2)
            test_name.append(item1)
            count_test_3 += 1
        elif item3 == 3 and count_dev_3 < dev_len:
            dev_label.append(item3)
            dev_text.append(item2)
            dev_name.append(item1)
            count_dev_3 += 1
        elif item3 == 3:
            train_label.append(item3)
            train_text.append(item2)
            train_name.append(item1)

        elif item3 == 2 and count_test_2 < test_len:
            test_label.append(item3)
            test_text.append(item2)
            test_name.append(item1)
            count_test_2 += 1
        elif item3 == 2 and count_dev_2 < dev_len:
            dev_label.append(item3)
            dev_text.append(item2)
            dev_name.append(item1)
            count_dev_2 += 1
        elif item3 == 2:
            train_label.append(item3)
            train_text.append(item2)
            train_name.append(item1)

        elif item3 == 1 and count_test_1 < test_len:
            test_label.append(item3)
            test_text.append(item2)
            test_name.append(item1)
            count_test_1 += 1
        elif item3 == 1 and count_dev_1 < dev_len:
            dev_label.append(item3)
            dev_text.append(item2)
            dev_name.append(item1)
            count_dev_1 += 1
        elif item3 == 1:
            train_label.append(item3)
            train_text.append(item2)
            train_name.append(item1)

        elif item3 == 0 and count_test_0 < test_len:
            test_label.append(item3)
            test_text.append(item2)
            test_name.append(item1)
            count_test_0 += 1
        elif item3 == 0 and count_dev_0 < dev_len:
            dev_label.append(item3)
            dev_text.append(item2)
            dev_name.append(item1)
            count_dev_0 += 1
        elif item3 == 0:
            train_label.append(item3)
            train_text.append(item2)
            train_name.append(item1)

        train_df = {'name': train_name, 'text': train_text, 'label': train_label}
        train_df = pd.DataFrame(train_df)
        dev_df = {'name': dev_name, 'text': dev_text, 'label': dev_label}
        dev_df = pd.DataFrame(dev_df)
        test_df = {'name': test_name, 'text': test_text, 'label': test_label}
        test_df = pd.DataFrame(test_df)
        train_df.to_csv(train_path, index=False)
        dev_df.to_csv(dev_path, index=False)
        test_df.to_csv(test_path, index=False)


if __name__ == "__main__":
    df_1 = pd.read_csv('../data_source/data_1.csv')
    save_data_root_path = os.path.join('../data_done/data_1')
    shuffled_df_1 = df_1.sample(frac=1, random_state=42)
    #
    # # 分组信息
    # train_problem_1_group_path = os.path.join(save_data_root_path, 'problem_1_group_info_train.csv')
    # dev_problem_1_group_path = os.path.join(save_data_root_path, 'problem_1_group_info_dev.csv')
    # test_problem_1_group_path = os.path.join(save_data_root_path, 'problem_1_group_info_test.csv')
    # triple_classification(shuffled_df_1, '分组信息', train_problem_1_group_path, dev_problem_1_group_path,
    #                       test_problem_1_group_path)
    #
    # # 自尊水平
    # train_problem_1_esteem_path = os.path.join(save_data_root_path, 'problem_1_esteem_train.csv')
    # dev_problem_1_esteem_path = os.path.join(save_data_root_path, 'problem_1_esteem_dev.csv')
    # test_problem_1_esteem_path = os.path.join(save_data_root_path, 'problem_1_esteem_test.csv')
    # triple_classification(shuffled_df_1, '自尊水平', train_problem_1_esteem_path, dev_problem_1_esteem_path,
    #                       test_problem_1_esteem_path)
    #
    # # 焦虑情绪
    # train_problem_1_anxiety_path = os.path.join(save_data_root_path, 'problem_1_anxiety_train.csv')
    # dev_problem_1_anxiety_path = os.path.join(save_data_root_path, 'problem_1_anxiety_dev.csv')
    # test_problem_1_anxiety_path = os.path.join(save_data_root_path, 'problem_1_anxiety_test.csv')
    # triple_classification(shuffled_df_1, '焦虑情绪', train_problem_1_anxiety_path, dev_problem_1_anxiety_path,
    #                       test_problem_1_anxiety_path)
    #
    # # 家庭功能
    # train_problem_1_family_path = os.path.join(save_data_root_path, 'problem_1_family_train.csv')
    # dev_problem_1_family_path = os.path.join(save_data_root_path, 'problem_1_family_dev.csv')
    # test_problem_1_family_path = os.path.join(save_data_root_path, 'problem_1_family_test.csv')
    # triple_classification(shuffled_df_1, '家庭功能', train_problem_1_family_path, dev_problem_1_family_path,
    #                       test_problem_1_family_path, test_len=8, dev_len=5)
    #
    # # 压力大
    train_problem_1_pressure_path = os.path.join(save_data_root_path, 'problem_1_pressure_train.csv')
    dev_problem_1_pressure_path = os.path.join(save_data_root_path, 'problem_1_pressure_dev.csv')
    test_problem_1_pressure_path = os.path.join(save_data_root_path, 'problem_1_pressure_test.csv')
    binary_classification(shuffled_df_1, '压力大', train_problem_1_pressure_path, dev_problem_1_pressure_path,
                          test_problem_1_pressure_path)
    #
    # # 表达抑制 expression 3
    # train_problem_1_expression_path = os.path.join(save_data_root_path, 'problem_1_expression_train.csv')
    # dev_problem_1_expression_path = os.path.join(save_data_root_path, 'problem_1_expression_dev.csv')
    # test_problem_1_expression_path = os.path.join(save_data_root_path, 'problem_1_expression_test.csv')
    # triple_classification(shuffled_df_1, '表达抑制', train_problem_1_expression_path, dev_problem_1_expression_path,
    #                       test_problem_1_expression_path)
    #
    # # depression 抑郁情绪  5
    # train_problem_1_depression_path = os.path.join(save_data_root_path, 'problem_1_depression_train.csv')
    # dev_problem_1_depression_path = os.path.join(save_data_root_path, 'problem_1_depression_dev.csv')
    # test_problem_1_depression_path = os.path.join(save_data_root_path, 'problem_1_depression_test.csv')
    # triple_classification(shuffled_df_1, '抑郁情绪', train_problem_1_depression_path, dev_problem_1_depression_path,
    #                       test_problem_1_depression_path)
    #
    # # 学业负担 5
    # train_problem_1_study_path = os.path.join(save_data_root_path, 'problem_1_study_train.csv')
    # dev_problem_1_study_path = os.path.join(save_data_root_path, 'problem_1_study_dev.csv')
    # test_problem_1_study_path = os.path.join(save_data_root_path, 'problem_1_study_test.csv')
    # binary_classification(shuffled_df_1, '学业负担', train_problem_1_study_path, dev_problem_1_study_path,
    #                       test_problem_1_study_path)
    #
    # # 积极应对 positive 3
    # train_problem_1_positive_path = os.path.join(save_data_root_path, 'problem_1_positive_train.csv')
    # dev_problem_1_positive_path = os.path.join(save_data_root_path, 'problem_1_positive_dev.csv')
    # test_problem_1_positive_path = os.path.join(save_data_root_path, 'problem_1_positive_test.csv')
    # binary_classification(shuffled_df_1, '积极应对', train_problem_1_positive_path, dev_problem_1_positive_path,
    #                       test_problem_1_positive_path)
    #
    # # 心理韧性 3 toughness
    # train_problem_1_toughness_path = os.path.join(save_data_root_path, 'problem_1_toughness_train.csv')
    # dev_problem_1_toughness_path = os.path.join(save_data_root_path, 'problem_1_toughness_dev.csv')
    # test_problem_1_toughness_path = os.path.join(save_data_root_path, 'problem_1_toughness_test.csv')
    # triple_classification(shuffled_df_1, '心理韧性', train_problem_1_toughness_path, dev_problem_1_toughness_path,
    #                       test_problem_1_toughness_path)
    # 生活满意度
    train_problem_all_life_path = os.path.join(save_data_root_path, 'problem_1_life_train.csv')
    dev_problem_all_life_path = os.path.join(save_data_root_path, 'problem_1_life_dev.csv')
    test_problem_all_life_path = os.path.join(save_data_root_path, 'problem_1_life_test.csv')
    life_satisfactory(shuffled_df_1, train_problem_all_life_path, dev_problem_all_life_path,
                      test_problem_all_life_path)
