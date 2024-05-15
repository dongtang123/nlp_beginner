import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

data_root_path = os.path.join("/data/caojieming/nlp_beginner/data/")
data_test_path = os.path.join(data_root_path, "merged_data.tsv")
data_train_path = os.path.join(data_root_path, "train.tsv")
bert_path = os.path.join("/data/caojieming/models/bert-base-uncased")


def data_load():
    data = pd.read_csv(data_train_path, sep='\t', header=0)[:90000]
    data_test = pd.read_csv(data_train_path, sep='\t', header=0)[90001:]
    data_train_list = []
    data_validation_list = []
    data_test_list = []
    data_train, data_validation = train_test_split(data, train_size=16000, test_size=3200, random_state=42)
    data_train_texts = [item.lower() for item in data_train['Phrase']]
    data_train_labels = [item for item in data_train['Sentiment']]
    data_train_list.append(data_train_texts)
    data_train_list.append(data_train_labels)
    data_validation_texts = [item.lower() for item in data_validation['Phrase']]
    data_validation_labels = [item for item in data_validation['Sentiment']]
    data_validation_list.append(data_validation_texts)
    data_validation_list.append(data_validation_labels)
    data_test = data_test.sample(n=3200, random_state=42)
    data_test_texts = [item.lower() for item in data_test['Phrase']]
    data_test_labels = [item for item in data_test['Sentiment']]
    data_test_list.append(data_test_texts)
    data_test_list.append(data_test_labels)

    return data_train_list, data_validation_list, data_test_list


class MyDataset(Dataset):
    def __init__(self, data_list, max_len):
        self.texts = data_list[0]
        self.labels = data_list[1]
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.texts[index]
        label = int(self.labels[index])
        encode_pair = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_len,
                                     return_tensors='pt')
        input_ids = encode_pair['input_ids'].squeeze(0)
        attention_mask = encode_pair['attention_mask'].squeeze(0)
        token_type_ids = encode_pair['token_type_ids'].squeeze(0)
        return input_ids, attention_mask, token_type_ids, label

    def __len__(self):
        length = len(self.texts)
        return length
