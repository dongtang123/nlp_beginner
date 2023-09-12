import numpy as np
import os
import csv
import json

# load data

train_data_path = os.path.join("D:\\data\\nlp_beginner\\classification\\train.tsv")

train_data = []

with open(train_data_path) as f:
    train_data_csv = csv.reader(f, delimiter='\t')
    for index, line in enumerate(train_data_csv):
        if index == 0:
            continue
        elif index > 10000:
            break
        train_data.append(line)
ngram_set =set()
for train_data_item in train_data:
    text = train_data_item[2]
    text = text.replace(",", "").replace(".", "")
    text = text.lower()
    words = text.split()
    n_list = [1,2,3]
    for n in n_list:
        ngrams= [words[i:i+n] for i in range(len(words)-n+1)]
        for ngram in ngrams:
            ngram = tuple(ngram)
            ngram_set.add(ngram)
ngram_set_json = {"ngram_set": list(ngram_set)}
ngram_set_json_path = os.path.join("D:\\data\\nlp_beginner\\classification\\ngram_set_json.json")
with open(ngram_set_json_path, "w") as f:
    json.dump(ngram_set_json, f)
ngram_features_vector = np.zeros((len(train_data),len(ngram_set)))
train_label = []
for index,train_data_item in enumerate(train_data):
    label = int(train_data_item[3])
    train_label.append(label)
    text = train_data_item[2]
    text = text.replace(",", "").replace(".", "")
    text = text.lower()
    words = text.split()
    n_list = [1, 2, 3]
    for n in n_list:
        ngrams= [words[i:i+n] for i in range(len(words)-n+1)]
        for ngram in ngrams:
            ngram = tuple(ngram)
            if ngram in ngram_set:
                set_index = list(ngram_set).index(ngram)
                ngram_features_vector[index][set_index]+=1
ngram_features_vector = ngram_features_vector.tolist()
feature_vectors_json = {"vectors": ngram_features_vector,"label":train_label}
feature_vectors_path = os.path.join("D:\\data\\nlp_beginner\\classification\\ngram_feature_vectors_10000.json")
with open(feature_vectors_path, "w") as f:
    json.dump(feature_vectors_json, f)