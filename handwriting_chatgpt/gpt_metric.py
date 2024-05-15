from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score, f1_score
import pandas as pd
import os


def evaluation(path):
    df = pd.read_csv(path)
    list_true = [item for item in df['label']]
    list_pred = [item for item in df['predict_label']]
    acc = accuracy_score(list_true, list_pred)
    precision = precision_score(list_true, list_pred, average='macro')
    recall = recall_score(list_true, list_pred, average='macro')
    f1 = f1_score(list_true, list_pred, average='macro')
    report = classification_report(list_true, list_pred)
    print(f"acc is {acc:.2f}")
    print(f"precision is {precision:.2f}")
    print(f"recall is {recall:.2f}")
    print(f"f1 is {f1:.2f}")
    print(report)


if __name__ == "__main__":
    csv_path = os.path.join('./data/result/standard_5_anxiety_result.csv')
    evaluation(csv_path)
