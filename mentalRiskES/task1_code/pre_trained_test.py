from transformers import BertTokenizer, BertModel
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from transformers import pipeline
import torch


def pipline_cls(path):
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pipe = pipeline("text-classification", model=r"D:\nlp\bert\spanish-sentiment-model", device=0)
    df = pd.read_csv(path)
    text_list = [text for text in df['message']]
    results = []
    for text in text_list:
        res = pipe(text)
        for item in range(1, 6):
            if res[0]['label'][0] == str(item) and item >= 3:
                results.append(1)
            elif res[0]['label'][0] == str(item) and item <= 2:
                results.append(0)

    print(results)
    return results




if __name__ == "__main__":
    csv_fusion = os.path.join("../data/task1/trial/data_labeled_fusion.csv")
    csv_split = os.path.join("../data/task1/trial/data_labeled_split.csv")
    # pipline_cls(csv_fusion)
    pipline_cls(csv_split)
