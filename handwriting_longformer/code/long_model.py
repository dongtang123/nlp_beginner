# -*- coding: utf-8 -*-

import torch.nn as nn
from transformers import LongformerModel
from torch.utils.data import DataLoader, Dataset
import os


class LongformerCLSModel(nn.Module):
    def __init__(self, config):
        super(LongformerCLSModel, self).__init__()

        self.longformer = LongformerModel.from_pretrained(config.model_path)
        self.classifier = nn.Linear(self.longformer.config.hidden_size, config.num_labels)
        # self.drop = nn.Dropout(0.5)
        # self.classifier_2 = nn.Linear(128, config.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[0][:, 0, :]
        logits = self.classifier(cls_output)
        # drop = self.drop(logits)
        # logits = self.classifier_2(drop)
        return logits
