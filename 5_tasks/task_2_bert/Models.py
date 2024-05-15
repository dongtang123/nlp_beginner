import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import os
import torch

bert_path = os.path.join("/data/caojieming/models/bert-base-uncased")


class Config(object):
    def __init__(self):
        self.bert_hidden_size = 768
        self.lstm_hidden_size = 256
        self.rnn_hidden_size = 256
        self.num_layers = 2
        self.rnn_num_layers = 2
        self.drop_out = 0.3
        self.num_class = 5


class BertCNN(nn.Module):
    def __init__(self, hidden_size, num_filter, filter_size, num_class,
                 dropout_rate):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=num_filter, kernel_size=(k, hidden_size)) for k in filter_size]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filter * len(filter_size), num_class)

    def conv_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x)
        x = x.squeeze(3)
        size = x.size(2)
        x = F.max_pool1d(x, size)
        x = x.squeeze(2)
        return x

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x[0], x[1], x[2]
        hidden_out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               output_hidden_states=False)
        out = hidden_out.last_hidden_state.unsqueeze(1)
        out = torch.cat([self.conv_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.bert_hidden_size, config.lstm_hidden_size, config.num_layers, batch_first=True,
                            dropout=config.drop_out, bias=True, bidirectional=True)
        self.dropout = nn.Dropout(config.drop_out)
        self.fc = nn.Linear(config.lstm_hidden_size * 2, config.num_class)

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x[0], x[1], x[2]
        out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        output_hidden_states=False).last_hidden_state
        out, (last_hidden, cell_state) = self.lstm(out)
        last_hidden_Left, last_hidden_Right = last_hidden[-2], last_hidden[-1]
        out = torch.cat([last_hidden_Left, last_hidden_Right], dim=-1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class BertRnn(nn.Module):
    def __init__(self, config):
        super(BertRnn, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.rnn = nn.RNN(config.bert_hidden_size, config.rnn_hidden_size, config.rnn_num_layers, batch_first=True)
        self.dropout = nn.Dropout(config.drop_out)
        self.fc = nn.Linear(config.rnn_hidden_size, config.num_class)

    def forward(self, x):
        input_ids, attention_mask = x[0], x[1]
        out = self.bert(input_ids, attention_mask=attention_mask,
                        output_hidden_states=False).last_hidden_state[:, 0, :]
        output, hn = self.rnn(out)
        out = hn[-1]
        out = self.dropout(out)
        out = self.fc(out)
        return out
