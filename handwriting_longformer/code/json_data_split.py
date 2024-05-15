import pandas as pd
import json
import os


def list_split(list_0_key_all, list_0_value_all, list_1_key_all, list_1_value_all, start, end):
    list_text_train = [value[0]['text'][0] for value in list_0_value_all]
    list_name_train = [key for key in list_0_key_all]
    list_label_train = [0] * len(list_name_train)
    list_name_dev, list_text_dev, list_label_dev = [], [], []
    list_name_test, list_text_test, list_label_test = [], [], []
    for i in range(start, end):
        if len(list_name_test) < 12:
            list_name_test.append(list_1_key_all[i])
            list_text_test.append(list_1_value_all[i][0]['text'][0])
            list_label_test.append(list_1_value_all[i][1]['label'])
        else:
            list_name_dev.append(list_1_key_all[i])
            list_text_dev.append(list_1_value_all[i][0]['text'][0])
            list_label_dev.append(list_1_value_all[i][1]['label'])

    # for i in range(start, end):
    #     list_1_key_all.remove(list_1_key_all[i])
    #     list_1_value_all.remove(list_1_value_all[i])
    list_1_value = [] + list_1_value_all
    list_1_key = [] + list_1_key_all
    del list_1_key[start:end]
    del list_1_value[start:end]
    for i in range(0, len(list_1_key)):
        list_name_train += [list_1_key[i]] * len(list_1_value[i][0]['text'])
        list_label_train += [list_1_value[i][1]['label']] * len(list_1_value[i][0]['text'])
        list_text_train += list_1_value[i][0]['text']
    # print(len(list_text_train), len(list_name_train), len(list_label_train))
    # print(len(list_text_dev), len(list_name_dev), len(list_label_dev))
    # print(len(list_text_test), len(list_name_test), len(list_label_test))
    df_train = {'name': list_name_train[20:], 'text': list_text_train[20:], 'label': list_label_train[20:]}
    df_train = pd.DataFrame(df_train)

    df_dev = {'name': list_name_dev + list_name_train[:end - start - 12],
              'text': list_text_dev + list_text_train[:end - start - 12],
              'label': list_label_dev + list_label_train[:end - start - 12]}
    df_dev = pd.DataFrame(df_dev)

    df_test = {'name': list_name_test + list_name_train[end - start - 12:end - start],
               'text': list_text_test + list_text_train[end - start - 12:end - start],
               'label': list_label_test + list_label_train[end - start - 12:end - start]}
    df_test = pd.DataFrame(df_test)
    return df_train, df_dev, df_test


def split_distribution(data):
    list_0_key_all, list_0_value_all = [], []
    list_1_key_all, list_1_value_all = [], []
    for key, value in data.items():
        if value[1]['label'] == 0:
            list_0_key_all.append(key)
            list_0_value_all.append(value)
        else:
            list_1_key_all.append(key)
            list_1_value_all.append(value)

    df_train_1, df_dev_1, df_test_1 = list_split(list_0_key_all, list_0_value_all, list_1_key_all, list_1_value_all, 0,
                                                 20)
    tuple_1 = (df_train_1, df_dev_1, df_test_1)
    print(len(df_train_1['name']))

    df_train_2, df_dev_2, df_test_2 = list_split(list_0_key_all, list_0_value_all, list_1_key_all, list_1_value_all, 20,
                                                 40)
    tuple_2 = (df_train_2, df_dev_2, df_test_2)
    print(len(df_train_2['name']))

    df_train_3, df_dev_3, df_test_3 = list_split(list_0_key_all, list_0_value_all, list_1_key_all, list_1_value_all, 40,
                                                 58)
    tuple_3 = (df_train_3, df_dev_3, df_test_3)
    print(len(df_train_3['name']))

    list_temp = [tuple_1, tuple_2, tuple_3]
    return list_temp


def split_data(path1, save_path_root):
    with open(path1, 'r', encoding='utf8') as json_file:
        json_data = json.load(json_file)
    # print(json_data)
    list_temp = split_distribution(json_data)

    for index, item in enumerate(list_temp):
        print(index)
        item[0].to_csv(os.path.join(save_path_root, f'train_{index}.csv'), index=False)
        item[1].to_csv(os.path.join(save_path_root, f'dev_{index}.csv'), index=False)
        item[2].to_csv(os.path.join(save_path_root, f'test_{index}.csv'), index=False)


if __name__ == "__main__":
    problem_1_anxiety_all = os.path.join('../cross_validation_json/problem_1_anxiety_all.json')
    problem_1_depression_all = os.path.join('../cross_validation_json/problem_1_depression_all.json')
    problem_2_anxiety_all = os.path.join('../cross_validation_json/problem_2_anxiety_all.json')
    problem_2_depression_all = os.path.join('../cross_validation_json/problem_2_depression_all.json')

    problem_1_anxiety_save = os.path.join('../cross_validation_csv/problem_1_anxiety')
    problem_1_depression_save = os.path.join('../cross_validation_csv/problem_1_depression')
    problem_2_anxiety_save = os.path.join('../cross_validation_csv/problem_2_anxiety')
    problem_2_depression_save = os.path.join('../cross_validation_csv/problem_2_depression')

    split_data(problem_1_anxiety_all, problem_1_anxiety_save)
    # split_data(problem_1_depression_all, problem_1_depression_save)
    split_data(problem_2_anxiety_all, problem_2_anxiety_save)
    # split_data(problem_2_depression_all, problem_2_depression_save)
