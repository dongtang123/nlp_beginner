import numpy as np
import os
import csv
import json
import model_softmax

train_data_path_ngram = os.path.join("D:\\data\\nlp_beginner\\classification\\ngram_feature_vectors_10000.json")
train_data_path_bow = os.path.join("D:\\data\\nlp_beginner\\classification\\feature_vectors_10000.json")
test_data_path_ngram = os.path.join("D:\\data\\nlp_beginner\\classification\\ngram_test_10000.json")


def load_data(train_data_path):
    with open(train_data_path) as f:
        data = json.load(f)
    return data


# data_bow = load_ngram(train_data_path_bow)
data_ngram = load_data(train_data_path_ngram)
test_ngram = load_data(test_data_path_ngram)
X = data_ngram['vectors']
y = data_ngram['label']
x = test_ngram['vectors']
y_test = test_ngram['label']
model = model_softmax.softmax_regression()
w, b = model.train(feature_vectors=X, labels=y, lr=0.01, epochs=1000, num_classes=5)
w = w.tolist()
b = b.tolist()

parameters = {'w': w, 'b': b}
parameters_path = os.path.join("D:\\data\\nlp_beginner\\classification\\parameters.json")
with open(parameters_path, "w") as f:
    json.dump(parameters, f)
res = model.predict(x=x)
acc = 0
for p, l in zip(res, y_test):
    if p == l:
        acc += 1
acc /= len(y_test)
print(acc)
