import torch
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import random
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import jieba
from collections import Counter
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

hidden_size = 1200


def macro_metrics(y_true, y_pred):
    macro_acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    print("Macro-Accuracy:", macro_acc)
    print("Macro-Precision:", macro_precision)
    print("Macro-Recall:", macro_recall)
    print("Macro-F1:", macro_f1)
    # return macro_acc, macro_precision, macro_recall, macro_f1


def micro_metrics(y_true, y_pred):
    micro_acc = accuracy_score(y_true, y_pred)
    micro_precision = precision_score(y_true, y_pred, average='micro')
    micro_recall = recall_score(y_true, y_pred, average='micro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    print("Micro-Accuracy:", micro_acc)
    print("Micro-Precision:", micro_precision)
    print("Micro-Recall:", micro_recall)
    print("Micro-F1:", micro_f1)
    # return micro_acc, micro_precision, micro_recall, micro_f1


def read_data_split(path):
    train_content_list, test_content_list = [], []
    train_label_list, test_label_list = [], []
    df = pd.read_csv(path)
    for review, label in zip(df['review'], df['label']):
        train_content_list.append(review)
        train_label_list.append(label)
    random.seed(42)
    combined_list = list(zip(train_content_list, train_label_list))
    random.shuffle(combined_list)
    shuffle_content_list, shuffle_label_list = zip(*combined_list)
    train_content_list, test_content_list, train_label_list, test_label_list = train_test_split(shuffle_content_list,
                                                                                                shuffle_label_list,
                                                                                                test_size=0.1)
    return train_content_list, test_content_list, train_label_list, test_label_list


def get_vector_logistic(train, test):
    vectorizer = CountVectorizer()
    vector_train = vectorizer.fit_transform(train)
    vector_test = vectorizer.transform(test)

    # scaler = StandardScaler(with_mean=False)
    # vector_train = scaler.fit_transform(vector_train)
    # vector_test = scaler.transform(vector_test)

    svd = TruncatedSVD(n_components=hidden_size)
    vector_train = svd.fit_transform(vector_train)
    vector_test = svd.transform(vector_test)

    vector_train = torch.Tensor(vector_train)
    vector_test = torch.Tensor(vector_test)
    return vector_train, vector_test


class NaiveBayes:
    def __init__(self, content_list, label_list):
        self.label_train = torch.LongTensor(label_list)
        self.feature = content_list
        self.labels = torch.unique(self.label_train)
        self.feature_probability = {}
        self.class_probability = torch.zeros(len(self.labels))
        self.vocabulary = set()

    def train(self):
        for document in self.feature:
            self.vocabulary.update(list(jieba.cut(document)))
        for label in self.labels:
            self.class_probability[label] = torch.sum((self.label_train == label)).item() / len(self.label_train)
        for label in self.labels:
            flag = (self.label_train == label)
            documents_in_class = [self.feature[i] for i in range(len(self.feature)) if flag[i]]
            word_counts = Counter(word for document in documents_in_class for word in list(jieba.cut(document)))
            total_words_in_class = sum(word_counts.values())
            self.feature_probability[label.item()] = {word: count / total_words_in_class for word, count in
                                                      word_counts.items()}

    def predict(self, test):
        pred_list = []
        for document in test:
            scores = {c.item(): torch.log(self.class_probability[c]) for c in self.labels}
            for word in list(jieba.cut(document)):
                if word in self.vocabulary:
                    for label in self.labels:
                        scores[label.item()] += torch.log(
                            torch.tensor(self.feature_probability[label.item()].get(word, 1e-10)))
            pred = max(scores, key=scores.get)
            pred_list.append(pred)
        return pred_list


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        out = self.linear(data)
        out = self.sigmoid(out)

        return out


def train_logistic(train_data, train_label, test_data):
    vector_train, vector_test = get_vector_logistic(train_data, test_data)
    best_acc = 0.0
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_index, validation_index) in enumerate(kf.split(vector_train)):
        print(f"fold {fold}/10")
        train_label = torch.LongTensor(train_label)
        train_content_fold, validation_content_fold = vector_train[train_index].to(device), vector_train[
            validation_index].to(device)
        train_label_fold, validation_label_fold = train_label[train_index].to(device), train_label[validation_index].to(
            device)

        train_loader = DataLoader(TensorDataset(train_content_fold, train_label_fold), batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(validation_content_fold, validation_label_fold), batch_size=64,
                                shuffle=False)
        logistic_model = LogisticRegression().to(device)
        loss_func = nn.CrossEntropyLoss()
        opt = optim.SGD(logistic_model.parameters(), lr=0.01)
        for epoch in range(0, 10):
            logistic_model.train()
            total_loss = 0.0
            for inputs, labels in train_loader:
                opt.zero_grad()
                outputs = logistic_model(inputs).squeeze()
                loss = loss_func(outputs, labels.float())
                loss.backward()
                opt.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')
            logistic_model.eval()
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                outputs = logistic_model(inputs).squeeze()
                for pred, label in zip(outputs, labels):
                    if pred.item() < 0.5:
                        pred = torch.tensor(0)
                    else:
                        pred = torch.tensor(1)
                    if pred == label:
                        correct += 1
                total += len(labels)
            val_acc = correct / total
            print(f'Validation acc: {val_acc * 100:.2f}%')
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(logistic_model.state_dict(), 'best_logistic_model.pth')
                print('Best model saved.')


def test_logistic(train_data, test_data, test_label):
    vector_train, vector_test = get_vector_logistic(train_data, test_data)
    test_dataset = TensorDataset(vector_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    loaded_model = LogisticRegression()
    loaded_model.load_state_dict(torch.load('best_logistic_model.pth'))
    loaded_model.eval()
    pred_list = []
    for inputs in test_loader:
        outputs = loaded_model(inputs[0]).squeeze()
        for pred in outputs:
            if pred.item() <= 0.5:
                pred = torch.tensor(0)
            else:
                pred = torch.tensor(1)
            pred_list.append(pred.cpu().numpy())
    macro_metrics(pred_list, test_label)
    micro_metrics(pred_list, test_label)
    print(classification_report(pred_list, test_label))


def bayes_train_test(train_data, train_label, test_data, test_label):
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    best_acc = 0.0
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    best_model = None
    for fold, (train_index, validation_index) in enumerate(kf.split(train_data)):
        print(f"fold {fold}/10")
        train_content_fold, validation_content_fold = train_data[train_index], train_data[validation_index]
        train_label_fold, validation_label_fold = train_label[train_index], train_label[validation_index]
        naive_bayes = NaiveBayes(train_content_fold, train_label_fold)
        naive_bayes.train()
        pred = naive_bayes.predict(validation_content_fold)
        current_acc = accuracy_score(validation_label_fold, pred)
        print(f"current acc is{current_acc}")
        if current_acc > best_acc:
            best_acc = current_acc
            best_model = naive_bayes
    pred_class = best_model.predict(test_data)
    macro_metrics(pred_class, test_label)
    micro_metrics(pred_class, test_label)


if __name__ == "__main__":
    test_path = 'aug_problem_1_anxiety_test.csv'
    train_path = 'aug_problem_1_anxiety_train.csv'
    # train_data, test_data, train_label, test_label = read_data_split(csv_path)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    train_text = [item for item in df_train['text']]
    train_labels = [item for item in df_train['label']]
    test_text = [item for item in df_test['text']]
    test_labels = [item for item in df_test['label']]

    # print('Beginning of logistic regression:')
    # train_logistic(train_data, train_label, test_data)
    # print('Result of logistic regression:')
    # test_logistic(train_data, test_data, test_label)

    print('Result of logistic bayes:')
    bayes_train_test(train_text, train_labels, test_text, test_labels)
