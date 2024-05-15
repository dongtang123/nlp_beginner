import json
import os
import pandas as pd
import torch.cuda
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

train_path = "/data/caojieming/nlp_beginner/data/text_matching/snli_1.0/snli_1.0_train.jsonl"
dev_path = "/data/caojieming/nlp_beginner/data/text_matching/snli_1.0/snli_1.0_dev.jsonl"
test_path = "/data/caojieming/nlp_beginner/data/text_matching/snli_1.0/snli_1.0_test.jsonl"
bert_path = os.path.join("/data/caojieming/models/bert-base-uncased")


class MyDataset(Dataset):
    def __init__(self, data_list, max_len):
        self.sentence_1_list = data_list[0]
        self.sentence_2_list = data_list[1]
        self.label_list = data_list[2]
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

    def __getitem__(self, item):
        sentence_1 = self.sentence_1_list[item]
        sentence_2 = self.sentence_2_list[item]
        label = self.label_list[item]
        tokenizer_res_1 = self.tokenizer(sentence_1, padding='max_length', truncation=True, max_length=self.max_len,
                                         return_tensors="pt")
        tokenizer_res_2 = self.tokenizer(sentence_2, padding='max_length', truncation=True, max_length=self.max_len,
                                         return_tensors="pt")
        input_ids_1 = tokenizer_res_1['input_ids']
        input_ids_2 = tokenizer_res_2['input_ids']
        attention_mask_1 = tokenizer_res_1['attention_mask']
        attention_mask_2 = tokenizer_res_2['attention_mask']
        token_type_ids_1 = tokenizer_res_1['token_type_ids']
        token_type_ids_2 = tokenizer_res_2['token_type_ids']
        return input_ids_1, attention_mask_1, token_type_ids_1, input_ids_2, attention_mask_2, token_type_ids_2, label

    def __len__(self):
        length = len(self.sentence_1_list)
        return length


def get_data(path):
    # neutral : 0
    # contradiction : -1
    # entailment : 1
    # - : 2
    sentence_1 = []
    sentence_2 = []
    label = []
    data_list = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            sentence_1.append(data['sentence1'])
            sentence_2.append(data['sentence2'])
            if data['gold_label'] == 'neutral':
                label.append(0)
            elif data['gold_label'] == 'entailment':
                label.append(1)
            elif data['gold_label'] == 'contradiction':
                label.append(-1)
            else:
                label.append(2)
    data_list.append(sentence_1)
    data_list.append(sentence_2)
    data_list.append(label)
    return data_list


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # train_data = get_data(train_path)
    # dev_data = get_data(dev_path)
    test_data = get_data(dev_path)
    test_dataset = MyDataset(test_data, max_len=16)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    for index, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        print(batch[-1])
        if index > 4:
            exit(0)
