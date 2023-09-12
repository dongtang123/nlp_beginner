import torch
from gensim.models import Word2Vec
import csv
from torch import nn
from torchtext import data
from torch.utils.data import ConcatDataset, DataLoader
from torchtext.data import Field, TabularDataset, LabelField, Dataset
import os
import random
import pandas as pd

import re

test_data_path = os.path.join("D:\\data\\nlp_beginner\\classification\\merged_data.tsv")
train_data_path = os.path.join("D:\\data\\nlp_beginner\\classification\\train.tsv")



def data_load(train_data_path, test_data_path):
    TEXT = Field(sequential=True, lower=True, batch_first=True)
    LABEL = LabelField(batch_first=True)
    train_data = TabularDataset(
        path=train_data_path,
        format='tsv',
        skip_header=True,
        fields=[(None, None), (None, None), ('text', TEXT), ('label', LABEL)],
    )
    test_data = TabularDataset(
        path=test_data_path,
        format='tsv',
        fields=[('text', TEXT), ('label', LABEL)],
        skip_header=True,
    )
    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)
    # train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    return train_data, test_data, TEXT, LABEL


if __name__ == "__main__":
    train_data, test_data, TEXT, LABEL = data_load(train_data_path, test_data_path)

