import pandas as pd
import os
from sklearn.metrics import classification_report


def compare_human_gold(human, gold):
    df_human = pd.read_csv(human)
    list_human = [item for item in df_human['label']]
    df_gold = pd.read_csv(gold)
    list_gold = [item for item in df_gold['label']]
    print("human check label ")
    print(list_human)
    print("gold label ")
    print(list_gold)
    print("gold label is different from human check label")
    for index, item1 in enumerate(list_human):
        if item1 != list_gold[index]:
            print(df_human['id'][index] + "," + df_human['content1'][index] + df_human['content2'][index],
                  ",",df_gold['label'][index],",", df_human['label'][index])
    res = classification_report(list_gold, list_human)
    print(res)


if __name__ == "__main__":
    path_human = os.path.join('test_shuffle_human_check.csv')
    path_gold = os.path.join('test_shuffle_label.csv')
    compare_human_gold(path_human, path_gold)
