import pandas as pd
import os
from sklearn.model_selection import train_test_split
import random
from data_split_1 import life_satisfactory,triple_classification,binary_classification




if __name__ == "__main__":
    df_1 = pd.read_csv('../data_source/data_all.csv')
    save_data_root_path = os.path.join('../data_done/data_all')
    shuffled_df_1 = df_1.sample(frac=1, random_state=42)

    # # 分组信息 5
    # train_problem_all_group_path = os.path.join(save_data_root_path, 'problem_all_group_info_train.csv')
    # dev_problem_all_group_path = os.path.join(save_data_root_path, 'problem_all_group_info_dev.csv')
    # test_problem_all_group_path = os.path.join(save_data_root_path, 'problem_all_group_info_test.csv')
    # triple_classification(shuffled_df_1, '分组信息', train_problem_all_group_path, dev_problem_all_group_path,
    #                       test_problem_all_group_path)
    #
    # # 自尊水平 1
    # train_problem_all_esteem_path = os.path.join(save_data_root_path, 'problem_all_esteem_train.csv')
    # dev_problem_all_esteem_path = os.path.join(save_data_root_path, 'problem_all_esteem_dev.csv')
    # test_problem_all_esteem_path = os.path.join(save_data_root_path, 'problem_all_esteem_test.csv')
    # triple_classification(shuffled_df_1, '自尊水平', train_problem_all_esteem_path, dev_problem_all_esteem_path,
    #                       test_problem_all_esteem_path)
    #
    # # 焦虑情绪 5
    # train_problem_all_anxiety_path = os.path.join(save_data_root_path, 'problem_all_anxiety_train.csv')
    # dev_problem_all_anxiety_path = os.path.join(save_data_root_path, 'problem_all_anxiety_dev.csv')
    # test_problem_all_anxiety_path = os.path.join(save_data_root_path, 'problem_all_anxiety_test.csv')
    # triple_classification(shuffled_df_1, '焦虑情绪', train_problem_all_anxiety_path, dev_problem_all_anxiety_path,
    #                       test_problem_all_anxiety_path)
    #
    # # 家庭功能 4
    # train_problem_all_family_path = os.path.join(save_data_root_path, 'problem_all_family_train.csv')
    # dev_problem_all_family_path = os.path.join(save_data_root_path, 'problem_all_family_dev.csv')
    # test_problem_all_family_path = os.path.join(save_data_root_path, 'problem_all_family_test.csv')
    # triple_classification(shuffled_df_1, '家庭功能', train_problem_all_family_path, dev_problem_all_family_path,
    #                       test_problem_all_family_path, test_len=8, dev_len=5)
    #
    # # 压力大 5
    train_problem_all_pressure_path = os.path.join(save_data_root_path, 'problem_all_pressure_train.csv')
    dev_problem_all_pressure_path = os.path.join(save_data_root_path, 'problem_all_pressure_dev.csv')
    test_problem_all_pressure_path = os.path.join(save_data_root_path, 'problem_all_pressure_test.csv')
    binary_classification(shuffled_df_1, '压力大', train_problem_all_pressure_path, dev_problem_all_pressure_path,
                          test_problem_all_pressure_path)
    #
    # # 表达抑制 expression 3
    # train_problem_all_expression_path = os.path.join(save_data_root_path, 'problem_all_expression_train.csv')
    # dev_problem_all_expression_path = os.path.join(save_data_root_path, 'problem_all_expression_dev.csv')
    # test_problem_all_expression_path = os.path.join(save_data_root_path, 'problem_all_expression_test.csv')
    # triple_classification(shuffled_df_1, '表达抑制', train_problem_all_expression_path, dev_problem_all_expression_path,
    #                       test_problem_all_expression_path)
    #
    # # depression 抑郁情绪  5
    # train_problem_all_depression_path = os.path.join(save_data_root_path, 'problem_all_depression_train.csv')
    # dev_problem_all_depression_path = os.path.join(save_data_root_path, 'problem_all_depression_dev.csv')
    # test_problem_all_depression_path = os.path.join(save_data_root_path, 'problem_all_depression_test.csv')
    # triple_classification(shuffled_df_1, '抑郁情绪', train_problem_all_depression_path, dev_problem_all_depression_path,
    #                       test_problem_all_depression_path)
    #
    # # 学业负担 5
    # train_problem_all_study_path = os.path.join(save_data_root_path, 'problem_all_study_train.csv')
    # dev_problem_all_study_path = os.path.join(save_data_root_path, 'problem_all_study_dev.csv')
    # test_problem_all_study_path = os.path.join(save_data_root_path, 'problem_all_study_test.csv')
    # binary_classification(shuffled_df_1, '学业负担', train_problem_all_study_path, dev_problem_all_study_path,
    #                       test_problem_all_study_path)
    #
    # # 积极应对 positive 3
    # train_problem_all_positive_path = os.path.join(save_data_root_path, 'problem_all_positive_train.csv')
    # dev_problem_all_positive_path = os.path.join(save_data_root_path, 'problem_all_positive_dev.csv')
    # test_problem_all_positive_path = os.path.join(save_data_root_path, 'problem_all_positive_test.csv')
    # binary_classification(shuffled_df_1, '积极应对', train_problem_all_positive_path, dev_problem_all_positive_path,
    #                       test_problem_all_positive_path)
    #
    # # 心理韧性 3 toughness
    # train_problem_all_toughness_path = os.path.join(save_data_root_path, 'problem_all_toughness_train.csv')
    # dev_problem_all_toughness_path = os.path.join(save_data_root_path, 'problem_all_toughness_dev.csv')
    # test_problem_all_toughness_path = os.path.join(save_data_root_path, 'problem_all_toughness_test.csv')
    # triple_classification(shuffled_df_1, '心理韧性', train_problem_all_toughness_path, dev_problem_all_toughness_path,
    #                       test_problem_all_toughness_path)

    # 生活满意度 life
    train_problem_all_life_path = os.path.join(save_data_root_path, 'problem_all_life_train.csv')
    dev_problem_all_life_path = os.path.join(save_data_root_path, 'problem_all_life_dev.csv')
    test_problem_all_life_path = os.path.join(save_data_root_path, 'problem_all_life_test.csv')
    life_satisfactory(shuffled_df_1, train_problem_all_life_path, dev_problem_all_life_path,
                      test_problem_all_life_path)
