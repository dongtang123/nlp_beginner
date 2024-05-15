import pandas as pd
import os


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
        if item3 <= 1 and count_test_1 < test_len:
            test_label.append(0)
            test_text.append(item2)
            test_name.append(item1)
            count_test_1 += 1
        elif item3 <= 1 and count_dev_1 < dev_len:
            dev_label.append(0)
            dev_text.append(item2)
            dev_name.append(item1)
            count_dev_1 += 1
        elif item3 <= 1:
            train_label.append(0)
            train_text.append(item2)
            train_name.append(item1)

        elif item3 == 2 and count_test_0 < test_len:
            test_label.append(1)
            test_text.append(item2)
            test_name.append(item1)
            count_test_0 += 1
        elif item3 == 2 and count_dev_0 < dev_len:
            dev_label.append(1)
            dev_text.append(item2)
            dev_name.append(item1)
            count_dev_0 += 1
        elif item3 == 2:
            train_label.append(1)
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
    df_1 = pd.read_csv('../data_source/data_5.csv')
    save_data_root_path = os.path.join('../binary_data/data_5')
    shuffled_df_1 = df_1.sample(frac=1, random_state=42)

    train_problem_1_anxiety_path = os.path.join(save_data_root_path, 'binary_problem_5_anxiety_train.csv')
    dev_problem_1_anxiety_path = os.path.join(save_data_root_path, 'binary_problem_5_anxiety_dev.csv')
    test_problem_1_anxiety_path = os.path.join(save_data_root_path, 'binary_problem_5_anxiety_test.csv')
    binary_classification(shuffled_df_1, '焦虑情绪', train_problem_1_anxiety_path, dev_problem_1_anxiety_path,
                          test_problem_1_anxiety_path)
