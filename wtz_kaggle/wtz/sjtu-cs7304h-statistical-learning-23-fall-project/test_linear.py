import random

import numpy as np
import pickle
import torch
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import KFold
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# 加载数据
train_features = pickle.load(open('train_feature.pkl', 'rb'))
train_labels = np.load('train_labels.npy')

svd = TruncatedSVD(n_components=1024)  # k为目标维度
train_features = svd.fit_transform(train_features)

batch_size = 256

# 创建DataLoader


class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 600
best_val_accuracy = 0.0
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_tensor = torch.from_numpy(train_features).float()
y_tensor = torch.from_numpy(train_labels).long()
for fold, (train_index, val_index) in enumerate(kf.split(X_tensor)):
    print(f"Fold {fold + 1}:")

    # 划分数据集
    X_train, X_val = X_tensor[train_index].to(device), X_tensor[val_index].to(device)
    y_train, y_val = y_tensor[train_index].to(device), y_tensor[val_index].to(device)

    # 创建 DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # 实例化模型
    model = LinearModel(1024, 256, output_size=len(np.unique(train_labels))).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.2)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

        model.eval()
        correct = 0
        total = 0
        for inputs, labels in tqdm(val_loader):
            outputs = model(inputs)
            predicted = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print('Best model saved.')

loaded_model = LinearModel(1024, 256, output_size=len(np.unique(train_labels)))
loaded_model.load_state_dict(torch.load('best_model.pth'))
loaded_model.eval()

test_features = pickle.load(open('test_feature.pkl', 'rb'))

test_features = svd.transform(test_features)
predictions = []
test_tensor = torch.from_numpy(test_features).float()
test_dataset = TensorDataset(test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for inputs in tqdm(test_loader):
    outputs = loaded_model(inputs[0])
    predicted = torch.argmax(outputs, 1)
    predictions.extend(predicted.cpu().numpy())
df = pd.DataFrame({'ID': np.arange(len(predictions)), 'label': predictions})
df.to_csv('predictions.csv', index=False)
print('end saved')
