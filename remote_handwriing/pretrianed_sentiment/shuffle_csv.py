import pandas as pd
import os


def shuffle_data(path):
    df = pd.read_csv(path)
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    data_shuffled = df_shuffled[['id', 'content1', 'content2']]
    label_shuffled = df_shuffled[['id','label']]
    data_shuffled.to_csv('test_shuffle_without_label.csv',index=False)
    label_shuffled.to_csv('test_shuffle_label.csv', index=False)


if __name__ == "__main__":
    data_path = os.path.join('test_case.csv')
    # shuffle_data(data_path)
