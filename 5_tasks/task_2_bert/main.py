import os.path

import torch
from data_process import data_load, MyDataset
from transformers import BertModel
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
from Models import BertCNN, BertLSTM, BertRnn, Config
import torch.nn.functional as F

max_len = 16
hidden_size = 768
num_filter = 3
filter_size = [2, 3, 4]
num_class = 5
dropout_rate = 0.3
epochs = 5
data_train_list, data_validation_list, data_test_list = data_load()
data_train = MyDataset(data_train_list, max_len)
data_validation = MyDataset(data_validation_list, max_len)
data_test = MyDataset(data_test_list, max_len)

# labels_train = data_train.labels
# labels_train = torch.tensor(labels_train)
# class_counts = torch.bincount(labels_train)
# class_weights = 1.0 / class_counts.float()
# sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(labels_train), replacement=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
train_dataloader = DataLoader(data_train, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(data_validation, batch_size=64, shuffle=True)
test_dataloader = DataLoader(data_test, batch_size=64, shuffle=False)

save_best_cnn = os.path.join("/data/caojieming/nlp_beginner/task_2_bert/save_model/bertCnn")
save_best_lstm = os.path.join("/data/caojieming/nlp_beginner/task_2_bert/save_model/bertLstm")
save_best_rnn = os.path.join("/data/caojieming/nlp_beginner/task_2_bert/save_model/bertRNN")


def train(model, path):
    acc_min = 0
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    for epoch in range(epochs):
        loss_sum = 0
        model.train()
        for index, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            outputs = model(batch)
            labels = batch[-1]
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss
            if index % 30 == 0:
                print("{} epoch  {} train loss is {}".format(epoch, index, loss_sum / (index + 1)))
                model.eval()
                with torch.no_grad():
                    count = 0
                    loss_sum_validation = 0
                    len_all = 0
                    len_pred = 0
                    for idx, batch_validation in enumerate(validation_dataloader):
                        batch_validation = tuple(t.to(device) for t in batch_validation)
                        outputs_validation = model(batch_validation)
                        pred = torch.argmax(outputs_validation, dim=1)
                        labels_validation = batch_validation[-1]
                        loss_validation = F.cross_entropy(outputs_validation, labels_validation)
                        len_all += len(labels_validation)
                        len_pred += int((labels_validation == pred).sum())
                        loss_sum_validation += loss_validation
                        count += 1
                    acc_validation_loss = loss_sum_validation / count
                    acc = len_pred / len_all

                    if acc > acc_min:
                        acc_min = acc
                        torch.save(model.state_dict(), path)
                    print(
                        "epoch {} validation loss is {}, validation acc is {} best validation acc is {}".format(epoch,
                                                                                                                acc_validation_loss,
                                                                                                                acc,
                                                                                                                acc_min))
                model.train()


def load_model(path):
    model = None
    if path == save_best_cnn:
        model = BertCNN(hidden_size, num_filter, filter_size, num_class, dropout_rate).to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
    elif path == save_best_lstm:
        config = Config()
        model = BertLSTM(config).to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
    elif path == save_best_rnn:
        config = Config()
        model = BertRnn(config).to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
    return model


def test(path):
    if path == save_best_cnn:
        model_cnn_best = load_model(save_best_cnn)
        len_pred_right = 0
        len_all = 0
        with torch.no_grad():
            for index, batch in enumerate(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                outputs = model_cnn_best(batch)
                pred = torch.argmax(outputs, dim=1)
                labels = batch[-1]
                len_pred_right += int((pred == labels).sum())
                len_all += len(labels)
        acc = len_pred_right / len_all
        print("test acc is {}".format(acc))
    elif path == save_best_lstm:
        model_lstm_best = load_model(save_best_lstm)
        len_pred_right = 0
        len_all = 0
        with torch.no_grad():
            for index, batch in enumerate(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                outputs = model_lstm_best(batch)
                pred = torch.argmax(outputs, dim=1)
                labels = batch[-1]
                len_pred_right += int((pred == labels).sum())
                len_all += len(labels)
        acc = len_pred_right / len_all
        print("test acc is {}".format(acc))
    elif path == save_best_rnn:
        model_rnn_best = load_model(save_best_rnn)
        len_pred_right = 0
        len_all = 0
        with torch.no_grad():
            for index, batch in enumerate(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                outputs = model_rnn_best(batch)
                pred = torch.argmax(outputs, dim=1)
                labels = batch[-1]
                len_pred_right += int((pred == labels).sum())
                len_all += len(labels)
        acc = len_pred_right / len_all
        print("test acc is {}".format(acc))


if __name__ == "__main__":
    # cnn
    model_cnn = BertCNN(hidden_size, num_filter, filter_size, num_class, dropout_rate).to(device)
    train(model_cnn, save_best_cnn)
    test(save_best_cnn)

    # lstm
    # config = Config()
    # model_lstm = BertLSTM(config).to(device)
    # train(model_lstm, save_best_lstm)
    # test(save_best_lstm)

    # rnn
    # config = Config()
    # model_rnn = BertRnn(config).to(device)
    # train(model_rnn, save_best_rnn)
    # test(save_best_rnn)
