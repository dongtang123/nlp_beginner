import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# 实际标签和预测标签
y_true = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
y_pred = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]

# 计算混淆矩阵
C2 = confusion_matrix(y_true, y_pred, labels=[0, 1])
print(C2)  # 打印出来看看

# 绘制热力图
sns.set()
f, ax = plt.subplots()
sns.heatmap(C2, annot=True, ax=ax)  # 画热力图
ax.set_title('Confusion Matrix')  # 标题
ax.set_xlabel('Predicted')  # x轴
ax.set_ylabel('True')  # y轴
plt.show()
