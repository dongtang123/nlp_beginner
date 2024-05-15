import os
import torch
from torch.optim import AdamW
from transformers import BertTokenizer
import torch.nn as nn
from long_model import LongformerCLSModel
from long_dataloader import load_data
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,f1_score
from long_config import Config
import numpy as np


def dev(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions, true_labels = [], []
    for batch in data_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu())
            true_labels.extend(labels.cpu())
    acc = accuracy_score(true_labels, predictions)
    return acc


def test_best(model, data_loader, config):
    model.load_state_dict(torch.load(config.save_best))
    model.eval()
    predictions, true_labels = [], []
    for batch in data_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu())
            true_labels.extend(labels.cpu())
    acc = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions,average='macro')
    recall = recall_score(true_labels, predictions,average='macro')
    f1 = f1_score(true_labels, predictions,average='macro')
    print(f"best acc is {acc:.2f},best precision is {precision:.2f},best recall is {recall:.2f},best f1 is {f1:.2f}")
    report = classification_report(true_labels, predictions)
    print(report)

def test_last(model, data_loader, config):
    model.load_state_dict(torch.load(config.save_last))
    model.eval()
    predictions, true_labels = [], []
    for batch in data_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu())
            true_labels.extend(labels.cpu())
    acc = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions,average='macro')
    recall = recall_score(true_labels, predictions,average='macro')
    print([item.item() for item in true_labels])
    print([item.item() for item in predictions])
    f1 = f1_score(true_labels, predictions,average='macro')
    print(f"best acc is {acc:.2f},best precision is {precision:.2f},best recall is {recall:.2f},best f1 is {f1:.2f}")
    report = classification_report(true_labels, predictions)
    print(report)

def train(model, config, train_dataloader, dev_dataloader):
    model.train()
    optimizer = AdamW(model.parameters(), config.lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(config.epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        current_acc = dev(model, dev_dataloader)
        if best_acc < current_acc:
            best_acc = current_acc
            torch.save(model.state_dict(), config.save_best)
        if epoch + 1 == 10:
            torch.save(model.state_dict(), config.save_last)
        print(f"current acc is {current_acc}; best acc is {best_acc}")
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    config = Config()
    longModel = LongformerCLSModel(config)

    longModel = longModel.to(config.device)

    tokenizer = BertTokenizer.from_pretrained(config.model_path)
    train_loader, dev_loader, test_loader = load_data(tokenizer, config)

    train(longModel, config, train_loader, dev_loader)
    test_best(longModel, test_loader, config)
    test_last(longModel, test_loader, config)
