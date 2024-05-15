import torch
from transformers import  LongformerModel,BertTokenizer
from torch.utils.data import DataLoader, Dataset

# 准备数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
          truncation=True
        )

        return {
          'text': text,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'labels': torch.tensor(label, dtype=torch.long)
        }

# 设置模型和分词器
tokenizer = BertTokenizer.from_pretrained('D:\\nlp\\longformer\\longformer_zh')
model = LongformerModel.from_pretrained('D:\\nlp\\longformer\\longformer_zh')

# 你的数据：文本和标签
texts = ["长文本样例 1", "长文本样例 2"]  # 替换为你的文本
labels = [0, 1]  # 替换为你的标签

# 创建数据加载器
dataset = TextDataset(texts, labels, tokenizer, max_len=512)
data_loader = DataLoader(dataset, batch_size=2)

# 训练或评估模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for data in data_loader:
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    labels = data['labels'].to(device)

    # 前向传播
    outputs = model(input_ids, attention_mask=attention_mask)

    # 计算损失
    loss = outputs.loss
    print(loss)
