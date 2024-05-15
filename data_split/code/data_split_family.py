import pandas as pd
import os
from sklearn.model_selection import train_test_split
import random
from data_split_1 import life_satisfactory, triple_classification, binary_classification

if __name__ == "__main__":
    df_1 = pd.read_csv('../data_source/data_family.csv')
    save_data_root_path = os.path.join('../data_done/data_family')
    shuffled_df_1 = df_1.sample(frac=1, random_state=42)

    #  家庭功能
    train_problem_family_path = os.path.join(save_data_root_path, 'problem_family_train.csv')
    dev_problem_family_path = os.path.join(save_data_root_path, 'problem_family_dev.csv')
    test_problem_family_path = os.path.join(save_data_root_path, 'problem_family_test.csv')
    triple_classification(shuffled_df_1, '家庭功能', train_problem_family_path, dev_problem_family_path,
                          test_problem_family_path)
