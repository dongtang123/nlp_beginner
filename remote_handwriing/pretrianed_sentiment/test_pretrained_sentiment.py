from transformers import BertForSequenceClassification, AutoModelForSequenceClassification
from transformers import BertTokenizer, AutoTokenizer
from transformers import pipeline
import torch
import pandas as pd
import os
from sklearn.metrics import classification_report
import numpy as np


def read_csv(path):
    df = pd.read_csv(path)
    id_list, content1_list, content2_list, content_list, label_list = [], [], [], [], []
    for id, content1, content2, label in zip(df['id'], df['content1'], df['content2'], df['label']):
        id_list.append(id)
        content1_list.append(content1)
        content2_list.append(content2)
        label_list.append(label)
        content = content1 + content2
        content_list.append(content)
    return id_list, content1_list, content2_list, content_list, label_list


def xuyuan(content, label):
    tokenizer = BertTokenizer.from_pretrained(
        'D:\\nlp_beginner\\remote_handwriing\\pretrianed_sentiment\\xuyuan-trial-sentiment-bert-chinese')
    model = BertForSequenceClassification.from_pretrained(
        'D:\\nlp_beginner\\remote_handwriing\\pretrianed_sentiment\\xuyuan-trial-sentiment-bert-chinese')

    pred_list = []
    for item in content:
        output = model(torch.tensor([tokenizer.encode(item)]))
        pred1 = torch.nn.functional.softmax(output.logits,
                                            dim=-1)  # [none,disgust,happiness,like,fear,saddness,anger,suprise]
        pred1 = torch.argmax(pred1, dim=1)
        positive = [2, 3, 7]
        negative = [1, 4, 5, 6]
        if pred1 in positive:
            pred_list.append(0)
        elif pred1 in negative:
            pred_list.append(2)
        else:
            pred_list.append(1)
    res = classification_report(label, pred_list)
    print(res)


def Erlangshen(content, label):
    tokenizer = AutoTokenizer.from_pretrained(
        'D:\\nlp_beginner\\remote_handwriing\\pretrianed_sentiment\\Erlangshen-Roberta-110M-Sentiment')
    model = BertForSequenceClassification.from_pretrained(
        'D:\\nlp_beginner\\remote_handwriing\\pretrianed_sentiment\\Erlangshen-Roberta-110M-Sentiment')
    pred_list = []
    for item in content:
        output = model(torch.tensor([tokenizer.encode(item)]))
        pred1 = torch.nn.functional.softmax(output.logits,
                                            dim=-1)  # [none,disgust,happiness,like,fear,saddness,anger,suprise]

        if pred1[0][1] > 0.66:  #
            pred_list.append(0)
        elif pred1[0][0] > 0.66:
            pred_list.append(2)
        else:
            pred_list.append(1)
    res = classification_report(label, pred_list)
    print(res)


def pipeline_transformer(content, label):
    tokenizer = AutoTokenizer.from_pretrained(
        "D:\\nlp\\bert\\bert-base-chinese")
    model = AutoModelForSequenceClassification.from_pretrained(
        "D:\\nlp\\bert\\bert-base-chinese")
    sentiment_analysis_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    for item in content:
        result = sentiment_analysis_pipeline(item)
        print(result)
    # exit(0)
    #
    # res = classification_report(label, pred_list)
    # print(res)


def multilingual(content, label):
    tokenizer = AutoTokenizer.from_pretrained(
        'D:\\nlp_beginner\\remote_handwriing\\pretrianed_sentiment\\distilbert-base-multilingual-cased-sentiments-student')
    model = AutoModelForSequenceClassification.from_pretrained(
        'D:\\nlp_beginner\\remote_handwriing\\pretrianed_sentiment\\distilbert-base-multilingual-cased-sentiments-student')
    pred_list = []
    no_softmax = []
    have_softmax = []
    for item in content:
        output = model(torch.tensor([tokenizer.encode(item)]))
        pred = torch.nn.functional.softmax(output.logits,
                                           dim=-1)  # 0,1,2积极中性消极
        no_softmax.append(output.logits[0].tolist())# .detach().numpy()
        have_softmax.append(pred[0].tolist())
        pred1 = torch.argmax(pred, dim=1)

        if pred1 == 0 and pred[0][0] > 0.5:
            pred_list.append(0)
        elif pred1 == 2 and pred[0][2] > 0.5:
            pred_list.append(2)
        else:
            pred_list.append(1)
    res = classification_report(label, pred_list)
    print(res)
    print("true labels: ", label)
    print("predict labels: ", pred_list)
    print("no_softmax score:", no_softmax)
    print("softmax score:", have_softmax)


def print_info(id_list, list_in):
    for id, item in zip(id_list, list_in):
        print(item)


if __name__ == "__main__":
    data_path = os.path.join("D:\\nlp_beginner\\remote_handwriing\\pretrianed_sentiment\\test_shuffle_human_check.csv")
    id_list, content1_list, content2_list, content_list, label_list = read_csv(data_path)
    # print_info(id_list,content_list)
    multilingual(content_list, label_list)
