import numpy as np
import os
import csv
import json

# load data

test_data_path = os.path.join("D:\\data\\nlp_beginner\\classification\\test.tsv")
test_label_path = os.path.join("D:\\data\\nlp_beginner\\classification\\sampleSubmission.csv")
ngram_set_path = os.path.join("D:\\data\\nlp_beginner\\classification\\ngram_set_json.json")
test_data = []
test_label = []
with open(test_data_path) as f:
    test_data_csv = csv.reader(f, delimiter='\t')
    for index, line in enumerate(test_data_csv):
        if index == 0:
            continue
        elif index > 1000:
            break
        test_data.append(line)
with open(test_label_path) as f:
    test_label_csv = csv.reader(f, delimiter=',')
    for index, line in enumerate(test_label_csv):
        if index == 0:
            continue
        elif index > 1000:
            break
        test_label.append(int(line[1]))
ngram_set = set()
with open(ngram_set_path,'r') as f:
    ngram_json = json.load(f)
    ngram_tuple = [tuple(sublist) for sublist in ngram_json['ngram_set']]
    ngram_set = set(ngram_tuple)


ngram_features_vector = np.zeros((len(test_data),len(ngram_set)))

for index,test_data_item in enumerate(test_data):
    text = test_data_item[2]
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
feature_vectors_json = {"vectors": ngram_features_vector,"label":test_label}
feature_vectors_path = os.path.join("D:\\data\\nlp_beginner\\classification\\ngram_test_10000.json")
with open(feature_vectors_path, "w") as f:
    json.dump(feature_vectors_json, f)