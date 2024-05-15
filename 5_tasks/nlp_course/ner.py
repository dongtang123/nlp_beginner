import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torchcrf import CRF
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report


class FinanceSinaDataset(Dataset):
    def __init__(self, texts, labels, label_map, tokenizer, max_len=280):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = texts
        self.labels = labels
        self.label_map = label_map

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        word_labels = self.labels[idx]

        labels = [self.label_map[label] for label in word_labels] + [self.label_map["O"]] * (
                self.max_len - len(word_labels))

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels[:self.max_len], dtype=torch.long)
        }


class BertCrfForNER(nn.Module):
    def __init__(self, num_labels):
        super(BertCrfForNER, self).__init__()
        self.num_labels = num_labels
        self.bert = BertForTokenClassification.from_pretrained('D:\\nlp\\bert\\bert-base-chinese',
                                                               num_labels=self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.byte(), reduction='mean')
            return loss
        else:
            return self.crf.decode(logits, mask=attention_mask.byte())


def create_data_loader(texts, labels, label_map, tokenizer, batch_size):
    dataset = FinanceSinaDataset(texts, labels, label_map, tokenizer)
    return DataLoader(dataset, batch_size=batch_size)


def load_and_split_data(filename, split_rate=(0.7, 0.15, 0.15)):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
        texts = [entry['text'] for entry in data]
        labels = [entry['labels'] for entry in data]

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels,
                                                                          test_size=split_rate[1] + split_rate[2])
    dev_texts, test_texts, dev_labels, test_labels = train_test_split(temp_texts, temp_labels,
                                                                      test_size=split_rate[2] / (
                                                                              split_rate[1] + split_rate[2]))

    return (train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels)


def train_model(train_loader, dev_loader, model, optimizer, device, epochs=10):
    model = model.to(device)
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss, total_accuracy = 0, 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        current_acc = evaluate_model(model, dev_loader, device)[0]

        print(
            f"Epoch {epoch + 1}/{epochs} train loss: {train_loss:.4f} | validation accuracy: {current_acc:.4f}")

        if current_acc > best_acc:
            torch.save(model, 'best_model_ner.pth')
            best_acc = current_acc


def macro_metrics(y_true, y_pred):
    macro_acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    return macro_acc, macro_precision, macro_recall, macro_f1


def micro_metrics(y_true, y_pred):
    micro_acc = accuracy_score(y_true, y_pred)
    micro_precision = precision_score(y_true, y_pred, average='micro')
    micro_recall = recall_score(y_true, y_pred, average='micro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    return micro_acc, micro_precision, micro_recall, micro_f1


def evaluate_model(model, data_loader, device):
    model.eval()
    model.to(device)
    pred_list = []
    true_list = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            predictions = model(input_ids, attention_mask)
            for i in range(len(labels)):
                label = labels[i][attention_mask[i] == 1]
                true_list.extend(label.tolist())
                prediction = predictions[i][:len(label)]
                pred_list.extend(prediction)
    macro_acc, macro_precision, macro_recall, macro_f1 = macro_metrics(true_list, pred_list)
    return macro_acc, macro_precision, macro_recall, macro_f1


def predict(model, tokenizer, text, label_map, device, max_len=280):
    model.eval()
    model.to(device)

    index_to_label = {value: key for key, value in label_map.items()}

    encoded_input = tokenizer(text, add_special_tokens=True, return_tensors="pt", max_length=max_len,
                              padding='max_length', truncation=True)
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    with torch.no_grad():
        predictions = model(input_ids, attention_mask)

    predicted_label_indices = predictions[0][:-2]
    predicted_labels = [index_to_label.get(idx, "O") for idx in predicted_label_indices]

    return predicted_labels


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained('D:\\nlp\\bert\\bert-base-chinese')
    model = BertCrfForNER(num_labels=9)
    (train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels) = load_and_split_data(
        'finance_sina.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    label_map = {"O": 0, "B-GPE": 1, "I-GPE": 2, "B-ORG": 3, "I-ORG": 4, "B-PER": 5, "I-PER": 6, "B-LOC": 7, "I-LOC": 8}

    train_loader = create_data_loader(train_texts, train_labels, label_map, tokenizer, batch_size=16)
    dev_loader = create_data_loader(dev_texts, dev_labels, label_map, tokenizer, batch_size=16)
    test_loader = create_data_loader(test_texts, test_labels, label_map, tokenizer, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    train_model(train_loader, dev_loader, model, optimizer, device, epochs=10)

    model_path = 'best_model_ner.pth'
    model_saved = torch.load(model_path)
    macro_acc, macro_precision, macro_recall, macro_f1 = evaluate_model(model_saved, test_loader, device)
    print(f"Test Macro-Accuracy:{macro_acc:.4}", )
    print(f"Test Macro-Precision:{macro_precision:.4}")
    print(f"Test Macro-Recall:{macro_recall:.4}")
    print(f"Test Macro-F1:{macro_f1:.4}")

    text_predict = "这是四川省和重庆市。"
    pred = predict(model_saved, tokenizer, text_predict, label_map, device, max_len=280)
    print(text_predict)
    print("predicted sequence is:\n", pred)
