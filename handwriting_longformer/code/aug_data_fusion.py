import pandas as pd
import os


def dict_info_train(path1, path2, path3):
    df_1 = pd.read_csv(path1)
    df_2 = pd.read_csv(path2)
    my_dict = {}
    for key in df_1['name']:
        my_dict[key] = ([], [], [])
    for key, value1, value2 in zip(df_1['name'], df_1['text'], df_1['label']):
        my_dict[key][0].append(value1)
        my_dict[key][2].append(value2)
    for key, value1 in zip(df_2['name'], df_2['text']):
        my_dict[key][1].append(value1)
    name_list, text_list, label_list = [], [], []
    for key, value in my_dict.items():
        long_length = len(value[0])
        list_2 = [value[1][i % len(value[1])] for i in range(long_length)]
        for item1, item2, item3 in zip(value[0], list_2, value[2]):
            name_list.append(key)
            text_list.append(item1 + '[SEP]' + item2)
            label_list.append(item3)
    df = pd.DataFrame({'name': name_list, 'text': text_list, 'label': label_list})
    df.to_csv(path3, index=False)


def data_fusion_test_dev(path1, path2, path3):
    df_1 = pd.read_csv(path1)
    df_2 = pd.read_csv(path2)
    name_list, text_list, label_list = [], [], []
    for item1, item2, item3, item4, item5 in zip(df_1['name'], df_2['name'], df_1['text'], df_2['text'], df_2['label']):
        if item1 == item2:
            name_list.append(item1)
            text_list.append(item3 + '[SEP]' + item4)
            label_list.append(item5)
    df = pd.DataFrame({'name': name_list, 'text': text_list, 'label': label_list})
    df.to_csv(path3, index=False)
    return


if __name__ == "__main__":
    # path_1 = os.path.join('../data/binary_data/aug_data/binary_aug_problem_1_anxiety_train.csv')
    # path_2 = os.path.join('../data/binary_data/aug_data/binary_aug_problem_2_anxiety_train.csv')
    # path_3 = os.path.join('../data/binary_data/aug_data/binary_aug_problem_1_2_anxiety_train.csv')
    # dict_info_train(path_1, path_2, path_3)
    path_1 = os.path.join('../data/binary_data/aug_data/binary_aug_problem_1_depression_dev.csv')
    path_2 = os.path.join('../data/binary_data/aug_data/binary_aug_problem_2_depression_dev.csv')
    path_3 = os.path.join('../data/binary_data/aug_data/binary_aug_problem_1_2_depression_dev.csv')
    data_fusion_test_dev(path_1, path_2, path_3)
