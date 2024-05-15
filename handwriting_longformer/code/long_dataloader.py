from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from transformers import BertTokenizer, LongformerModel,LongformerTokenizer
import os


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def read_data(path):
    df = pd.read_csv(path)
    text_list = [item for item in df['text']]
    label_list = [item for item in df['label']]
    return text_list, label_list


def load_data(tokenizer, config):
    train_text, train_label = read_data(config.train_path)
    dev_text, dev_label = read_data(config.dev_path)
    test_text, test_label = read_data(config.test_path)
    train_dataset = TextDataset(train_text, train_label, tokenizer, config.max_len)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)

    dev_dataset = TextDataset(dev_text, dev_label, tokenizer, config.max_len)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size)

    test_dataset = TextDataset(test_text, test_label, tokenizer, config.max_len)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    return train_loader, dev_loader, test_loader


