import pandas as pd
import os


def df_fusion_function(df_aug, df_normal, df_dev_normal, df_test_normal, path):
    name_list, text_list, label_list = [], [], []
    for item1, item2, item3 in zip(df_aug['name'], df_aug['text'], df_aug['label']):
        if item3 != 1:
            name_list.append(item1)
            text_list.append(item2)
            label_list.append(item3)
    for item1, item2, item3 in zip(df_normal['name'], df_normal['text'], df_normal['label']):
        if item3 == 1:
            name_list.append(item1)
            text_list.append(item2)
            label_list.append(0)
    for item1, item2, item3 in zip(df_dev_normal['name'], df_dev_normal['text'], df_dev_normal['label']):
        if item3 == 1:
            name_list.append(item1)
            text_list.append(item2)
            label_list.append(0)
    for item1, item2, item3 in zip(df_test_normal['name'], df_test_normal['text'], df_test_normal['label']):
        if item3 == 1:
            name_list.append(item1)
            text_list.append(item2)
            label_list.append(0)

    df_train = pd.DataFrame({'name': name_list, 'text': text_list, 'label': label_list})
    df_train.to_csv(path, index=False)


if __name__ == "__main__":
    df_train_aug = pd.read_csv('../data/aug_data/aug_problem_1_depression_train.csv')
    df_train_normal = pd.read_csv('../data/data_1/problem_1_depression_train.csv')
    df_dev = pd.read_csv('../data/aug_data/aug_problem_1_depression_dev.csv')
    df_test = pd.read_csv('../data/aug_data/aug_problem_1_depression_test.csv')

    save_data_root_path = os.path.join('../data/binary_data/aug_data/binary_aug_problem_1_depression_train.csv')
    # df_fusion_function(df_train_aug, df_train_normal, df_dev, df_test, save_data_root_path)
