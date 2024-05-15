import pandas as pd

df1 = pd.read_csv('../data_done/data_3/problem_3_anxiety_test.csv')
df2 = pd.read_csv('../data_done/data_3/problem_3_anxiety_train.csv')

# 执行左连接操作，保留df1中name包含但df2中user_id不包含的元素
result = pd.merge(df1, df2, left_on='name', right_on='name', how='left', indicator=True)
result = result[result['_merge'] == 'left_only']


# 提取结果中的name列
desired_elements = result['name'].tolist()

print(len(desired_elements))
