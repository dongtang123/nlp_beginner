import numpy as np
import os
import csv
import json

# load data
test_label_path = os.path.join("D:\\data\\nlp_beginner\\classification\\sampleSubmission.csv")
train_data_path = os.path.join("D:\\data\\nlp_beginner\\classification\\train.tsv")
test_data_path = os.path.join("D:\\data\\nlp_beginner\\classification\\test.tsv")
test_label = []
test_data = []
train_data = []
with open(test_label_path) as f:
    test_label_csv = csv.reader(f, skipinitialspace=True)

    for index, line in enumerate(test_label_csv):
        if index == 0:
            continue
        test_label.append(line)
with open(test_data_path) as f:
    test_data_csv = csv.reader(f, delimiter='\t')
    for index, line in enumerate(test_data_csv):
        if index == 0:
            continue
        test_data.append(line)

with open(train_data_path) as f:
    train_data_csv = csv.reader(f, delimiter='\t')
    for index, line in enumerate(train_data_csv):
        if index == 0:
            continue
        elif index>10000:
            break
        train_data.append(line)

# construct vocab
vocab = set()
for index, train_data_item in enumerate(train_data):
    text = train_data_item[2]
    text = text.replace(",", "").replace(".", "")
    text = text.lower()
    words = text.split()
    vocab.update(words)


# construct a vector of size len_train_data*len_vocab
len_vocab = len(vocab)
len_train_data = len(train_data)

feature_vectors = np.zeros((len_train_data, len_vocab))

# extract feature
train_label = []
for i, train_data_item in enumerate(train_data):
    text = train_data_item[2]
    label = train_data_item[3]
    text = text.replace(",", "").replace(".", "")
    words = text.split()
    for word in words:
        if word in vocab:
            index = list(vocab).index(word)
            feature_vectors[i, index] += 1
    train_label.append(label)
feature_vectors_list = feature_vectors.tolist()
feature_vectors_json = {"vectors": feature_vectors_list,"label":train_label}
feature_vectors_path = os.path.join("D:\\data\\nlp_beginner\\classification\\feature_vectors_10000.json")
with open(feature_vectors_path, "w") as f:
    json.dump(feature_vectors_json, f)
