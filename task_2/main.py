import data
import dataIterator
import model
import torch
import os

train_data_path = os.path.join("D:\\data\\nlp_beginner\\classification\\train.tsv")
test_data_path = os.path.join("D:\\data\\nlp_beginner\\classification\\merged_data.tsv")
batch_size = 32
shuffle = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data, test_data, TEXT, LABEL = data.data_load(train_data_path, test_data_path)
train_iter, dev_iter = dataIterator.iterator_construct_train(train_data, device)
test_iter = dataIterator.iterator_construct_test(test_data, device)

save_best_cnn = os.path.join("D:\\nlp_beginner\\task_2\\res_model\\" + "" + "textcnn.pth")
vocab_size = len(TEXT.vocab.itos)
embedding_dim = 256
num_filter = 200
filter_size = [2, 3, 4]
num_class = 5
dropout_rate = 0.1
learning_rate = 0.001
epochs = 10


def train():
    cnn = model.TextCNN(vocab_size, embedding_dim, num_filter, filter_size, num_class, dropout_rate)
    cnn.to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()
    acc_min = 0
    for epoch in range(epochs):
        cnn.train()
        loss_sum = 0
        for idx, batch in enumerate(train_iter):
            inputs, labels = batch.text, batch.label
            outputs = cnn(inputs)
            loss = loss_function(outputs, labels)
            cnn.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss
        cnn.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for idx, batch in enumerate(train_iter):
                inputs, labels = batch.text, batch.label
                outputs = cnn(inputs)
                pred = torch.argmax(outputs, dim=1)
                correct += (pred == labels).sum()
                total += len(labels)
            print('Epoch {}, train Accuracy {} train loss_total {}'.format(epoch, correct / total, loss_sum))
        cnn.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for idx, batch in enumerate(dev_iter):
                inputs, labels = batch.text, batch.label
                outputs = cnn(inputs)
                pred = torch.argmax(outputs, dim=1)
                correct += (pred == labels).sum()
                total += len(labels)
            acc = correct / total
            if acc > acc_min:
                acc_min = acc
                torch.save(cnn.state_dict(), save_best_cnn)
            print('Epoch {}, validation Accuracy {}'.format(epoch, correct / total))


def predict():
    cnn = model.TextCNN(vocab_size, embedding_dim, num_filter, filter_size, num_class, dropout_rate)
    cnn.to(device)
    cnn.load_state_dict(torch.load(save_best_cnn))
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for idx, batch in enumerate(test_iter):
            inputs, labels = batch.text, batch.label
            outputs = cnn(inputs)
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum()
            total += len(labels)
        acc = correct / total
        print('test Accuracy {}'.format(correct / total))


if __name__ == "__main__":
    train()
    predict()
