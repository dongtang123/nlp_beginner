import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filter, filter_size, num_class,
                 dropout_rate):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filter, (k, embedding_dim), padding=(k - 1, 0))
            for k in filter_size
        ])
        self.fc = nn.Linear(len(filter_size) * num_filter, num_class)
        self.dropout = nn.Dropout(dropout_rate)
    def conv_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x)
        x = x.squeeze(3)
        size = x.size(2)
        x = F.max_pool1d(x, size)
        x = x.squeeze(2)
        return x

    def forward(self, text):
        embedding = self.embedding(text).unsqueeze(1)
        out = torch.cat([self.conv_pool(embedding, conv) for conv in self.convs], 1)
        dropout = self.dropout(out)
        output = self.fc(dropout)
        return output
