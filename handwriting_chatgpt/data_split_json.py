import os
import random
from collections import defaultdict
import json
def split_data(dataPath, dev_size=8, test_size=12, num_splits=5):
    # 按标签分组数据
    with open(dataPath, 'r', encoding='utf8') as json_file:
        data_dict = json.load(json_file)

    grouped_data = defaultdict(list)
    for key, (text, label) in data_dict.items():
        grouped_data[label].append((key, text, label))

    # 确保每个标签组有足够的数据
    for label, items in grouped_data.items():
        if len(items) < dev_size * num_splits + test_size * num_splits:
            raise ValueError(f"Not enough data for label {label}.")

    # 对每个标签的数据进行随机打乱
    for label in grouped_data:
        random.shuffle(grouped_data[label])

    # 生成五组数据集
    splits = []
    for i in range(num_splits):
        train, dev, test = [], [], []
        for label, items in grouped_data.items():
            # 计算每个标签的dev和test的开始索引
            dev_start = i * dev_size
            test_start = len(items) - test_size * (num_splits - i)

            # 提取dev和test数据
            dev.extend(items[dev_start:dev_start + dev_size])
            test.extend(items[test_start:test_start + test_size])

            # 提取train数据
            train.extend(items[:dev_start] + items[dev_start + dev_size:test_start] + items[test_start + test_size:])

        splits.append((train, dev, test))

    return splits

if __name__ =="__main__":
    path = os.path.join(r'D:\nlp_beginner\handwriting_chatgpt\aug_data\data_1_test\problem_1_depression_all.json')

    split_data(path)