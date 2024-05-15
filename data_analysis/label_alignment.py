import pandas as pd
import os


def label_fusion(csv_path, label_path, save_path):
    csv_data = pd.read_csv(csv_path)
    excel_data = pd.read_csv(label_path)
    csv_data['name'] = csv_data['name'].astype(str)
    excel_data['user_id'] = excel_data['user_id'].astype(int)
    excel_data['user_id'] = excel_data['user_id'].astype(str)
    merged_data = pd.merge(csv_data, excel_data, left_on='name', right_on='user_id')
    merged_data.to_csv(save_path, index=False)


if __name__ == "__main__":
    label_path = os.path.join('label.csv')
    problem_1_path = os.path.join('data/problem_01.csv')
    problem_2_path = os.path.join('data/problem_02.csv')
    problem_3_path = os.path.join('data/problem_03.csv')
    problem_4_path = os.path.join('data/problem_04.csv')
    problem_5_path = os.path.join('data/problem_05.csv')
    problem_past_future_path = os.path.join('data/problem_past_future.csv')
    problem_all = os.path.join('data/problem_all.csv')
    problem_except_2 = os.path.join('data/problem_except_2.csv')
    problem_1_5_path = os.path.join('data/problem_1_5.csv')
    problem_study_path = os.path.join('data/problem_study.csv')
    problem_family_path = os.path.join('data/problem_family.csv')

    label_fusion(problem_1_path, label_path, 'data/data_1.csv')
    label_fusion(problem_2_path, label_path, 'data/data_2.csv')
    label_fusion(problem_3_path, label_path, 'data/data_3.csv')
    label_fusion(problem_4_path, label_path, 'data/data_4.csv')
    label_fusion(problem_5_path, label_path, 'data/data_5.csv')
    label_fusion(problem_past_future_path, label_path, 'data/data_past_future.csv')
    label_fusion(problem_all,label_path,'data/data_all.csv')
    label_fusion(problem_except_2,label_path,'data/data_all.csv')
    label_fusion(problem_1_5_path, label_path, 'data/data_1_5.csv')
    label_fusion(problem_study_path, label_path, 'data/data_study.csv')
    label_fusion(problem_family_path, label_path, 'data/data_family.csv')
