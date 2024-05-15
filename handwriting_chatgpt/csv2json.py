import pandas as pd
import json
import os


def data_from_csv(df):
    name_list = [item for item in df['name']]
    text_list = [item for item in df['text']]
    label_list = [item for item in df['label']]
    return name_list, text_list, label_list


def data_fusion(path1, path2, path3):
    df_1 = pd.read_csv(path1)
    df_2 = pd.read_csv(path2)
    df_3 = pd.read_csv(path3)
    name_list_1, text_list_1, label_list_1 = data_from_csv(df_1)
    name_list_2, text_list_2, label_list_2 = data_from_csv(df_2)
    name_list_3, text_list_3, label_list_3 = data_from_csv(df_3)
    name_list = name_list_1 + name_list_2 + name_list_3
    text_list = text_list_1 + text_list_2 + text_list_3
    label_list = label_list_1 + label_list_2 + label_list_3
    return name_list, text_list, label_list


def read_csv2json(path1, path2, path3, save_path):
    name_list, text_list, label_list = data_fusion(path1, path2, path3)
    data_dict = {key: ({'text': []}, {'label': label}) for key, label in zip(name_list, label_list)}
    for name, text in zip(name_list, text_list):
        data_dict[name][0]['text'].append(text)

    for key,value in data_dict.items():
        if value[1]['label'] == 0:
            value[0]['text'] = value[0]['text'][:1]
        else:
            value[0]['text'] = value[0]['text'][:5]
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    aug_problem_1_depression_test_dev = os.path.join('aug_data/data_1_test/aug_problem_1_depression_test_dev.csv')
    aug_problem_1_depression_train = os.path.join('aug_data/data_1_test/aug_problem_1_depression_train.csv')
    problem_1_depression_train = os.path.join('aug_data/data_1_test/problem_1_depression_train.csv')
    save_path_1 = os.path.join('aug_data/data_1_test/problem_1_depression_all.json')
    read_csv2json(aug_problem_1_depression_test_dev, aug_problem_1_depression_train, problem_1_depression_train,
                  save_path_1)

    aug_problem_1_anxiety_test_dev = os.path.join('aug_data/data_1_test/aug_problem_1_anxiety_test_dev.csv')
    aug_problem_1_anxiety_train = os.path.join('aug_data/data_1_test/aug_problem_1_anxiety_train.csv')
    problem_1_anxiety_train = os.path.join('aug_data/data_1_test/problem_1_anxiety_train.csv')
    save_path_2 = os.path.join('aug_data/data_1_test/problem_1_anxiety_all.json')
    read_csv2json(aug_problem_1_anxiety_test_dev, aug_problem_1_anxiety_train, problem_1_anxiety_train,
                  save_path_2)

    aug_problem_2_depression_test_dev = os.path.join('aug_data/data_2_test/aug_problem_2_depression_test_dev.csv')
    aug_problem_2_depression_train = os.path.join('aug_data/data_2_test/aug_problem_2_depression_train.csv')
    problem_2_depression_train = os.path.join('aug_data/data_2_test/problem_2_depression_train.csv')
    save_path_3 = os.path.join('aug_data/data_2_test/problem_2_depression_all.json')
    read_csv2json(aug_problem_2_depression_test_dev, aug_problem_2_depression_train, problem_2_depression_train,
                  save_path_3)

    aug_problem_2_anxiety_test_dev = os.path.join('aug_data/data_2_test/aug_problem_2_anxiety_test_dev.csv')
    aug_problem_2_anxiety_train = os.path.join('aug_data/data_2_test/aug_problem_2_anxiety_train.csv')
    problem_2_anxiety_train = os.path.join('aug_data/data_2_test/problem_2_anxiety_train.csv')
    save_path_4 = os.path.join('aug_data/data_2_test/problem_2_anxiety_all.json')
    read_csv2json(aug_problem_2_anxiety_test_dev, aug_problem_2_anxiety_train, problem_2_anxiety_train,
                  save_path_4)

