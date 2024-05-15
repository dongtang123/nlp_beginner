import pandas as pd
import os
from sklearn.model_selection import train_test_split
import random
from data_split_1 import life_satisfactory, triple_classification, binary_classification

if __name__ == "__main__":
    df_1 = pd.read_csv('../data_source/data_study.csv')
    save_data_root_path = os.path.join('../data_done/data_study')
    shuffled_df_1 = df_1.sample(frac=1, random_state=42)

    #  学业负担
    train_problem_study_path = os.path.join(save_data_root_path, 'problem_study_train.csv')
    dev_problem_study_path = os.path.join(save_data_root_path, 'problem_study_dev.csv')
    test_problem_study_path = os.path.join(save_data_root_path, 'problem_study_test.csv')
    binary_classification(shuffled_df_1, '学业负担', train_problem_study_path, dev_problem_study_path,
                          test_problem_study_path)
