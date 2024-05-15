from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def calculate_micro_metrics(y_true, y_pred):
    micro_accuracy = accuracy_score(y_true, y_pred)
    micro_precision = precision_score(y_true, y_pred, average='micro')

    # 计算微观召回率
    micro_recall = recall_score(y_true, y_pred, average='micro')

    # 计算微观F1分数
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    return micro_accuracy, micro_precision, micro_recall, micro_f1

# Example usage:
y_true = [1, 0, 21, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 1, 0, 1, 0, 1, 0, 1, 0, 0]

micro_accuracy, micro_precision, micro_recall, micro_f1 = calculate_micro_metrics(y_true, y_pred)

print("Micro-Accuracy:", micro_accuracy)
print("Micro-Precision:", micro_precision)
print("Micro-Recall:", micro_recall)
print("Micro-F1:", micro_f1)