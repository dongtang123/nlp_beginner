from transformers import BertTokenizer
import pandas as pd

# test_content = "英语、语文、数学、学习委员、副班长，地理生物。[SEP]家庭：和谐，友爱、妈妈、爸爸、弟弟、婆婆，爷爷、幺妈，堂姐，表姐，堂妹，表妹，[SEP]同学：朋友，同桌刘语沛瑶，小学同学及幼儿园同学吴晋宇，樊怡约，李艺轩、互相请教.[SEP]健康：蔬菜、荤食、素食、跑步、吃饭、睡觉"
# tokenizer = BertTokenizer.from_pretrained('D:\\nlp\\bert\\bert-base-chinese')
# tokens = tokenizer.tokenize(test_content, padding=True,truncation=True, return_tensors='pt',max_length=128,add_special_tokens=True)
# print(tokens)


# 读取两个CSV文件为DataFrame
df1 = pd.read_csv('data/problem_01.csv')
df2 = pd.read_csv('label.csv')

# 执行左连接操作，保留df1中name包含但df2中user_id不包含的元素
result = pd.merge(df1, df2, left_on='name', right_on='user_id', how='left', indicator=True)
result = result[result['_merge'] == 'left_only']

# 提取结果中的name列
desired_elements = result['name'].tolist()

print(desired_elements)
