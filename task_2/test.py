
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, Iterator

# 定义模型
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, hidden_dim, output_dim, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)# 将text(32*seq_length)对应的索引作为输入转化为embedding(32*seq_length*embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (filter_size, embedding_dim))
            for filter_size in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [self.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        hidden = self.relu(self.fc(cat))
        output = self.out(hidden)
        return output

# 设置超参数
vocab_size = 10000  # 词汇表大小
embedding_dim = 100  # 词向量维度
num_filters = 100  # 滤波器数量
filter_sizes = [3, 4, 5]  # 滤波器尺寸
hidden_dim = 256  # 隐藏层维度
output_dim = 2  # 输出维度（二分类）
dropout = 0.3  # Dropout概率
batch_size = 32  # 批处理大小
epochs = 10  # 迭代次数

# 定义字段
TEXT = Field(sequential=True, lower=True, batch_first=True)
LABEL = Field(sequential=False, use_vocab=False)

# 加载数据集
train_data, test_data = TabularDataset.splits(
    path='data_path',
    train='train.csv',
    test='test.csv',
    format='csv',
    fields=[('text', TEXT), ('label', LABEL)],
    skip_header=True
)

# 构建词汇表
TEXT.build_vocab(train_data, max_size=vocab_size)

# 创建数据迭代器
train_iterator, test_iterator = Iterator.splits(
    (train_data, test_data),
    batch_sizes=(batch_size, batch_size),
    sort_key=lambda x: len(x.text),
    sort_within_batch=False,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# 创建模型实例
model = TextCNN(vocab_size, embedding_dim, num_filters, filter_sizes, hidden_dim, output_dim, dropout)

# 将模型移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
model.train()
for epoch in range(epochs):
    for batch in train_iterator:
        text = batch.text.to(device)
        label = batch.label.to(device)
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_iterator:
        text = batch.text.to(device)
        label = batch.label.to(device)
        output = model(text)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

accuracy = 100 * correct / total
print('测试集准确率: {:.2f}%'.format(accuracy))
