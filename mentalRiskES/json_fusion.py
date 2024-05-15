import os
import json
import pandas as pd


def data_split_csv(label_path, folder_path, output_file):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    df = pd.read_csv(label_path)
    label_list = [item for item in df['label']]
    merged_data = []  # 创建一个空列表，用于存储合并后的 JSON 数据
    merged_label = []
    merged_subject = []
    round_list = []
    for file, label in zip(json_files, label_list):
        file_path = os.path.join(folder_path, file)  # 构建完整的文件路径
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)  # 读取 JSON 文件并解析为 Python 对象
            for index, item in enumerate(data):
                merged_data.append(item)
                merged_label.append(label)
                merged_subject.append(file.split('.')[0])
                round_list.append(float(index+1))
    id_message_list = []
    message_list = []
    date_list = []
    for item in merged_data:
        id_message_list.append(item['id_message'])
        message_list.append(item['message'])
        date_list.append(item['date'])
    my_dict = {"merged_subject": merged_subject, "id_message": id_message_list, "message": message_list,
               "date": date_list,"round":round_list ,"label": merged_label}
    df_labeled = pd.DataFrame(my_dict)
    df_labeled.to_csv(output_file, index=False)


def data_fusion_csv(label_path, folder_path, output_file):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    df = pd.read_csv(label_path)
    label_list = [item for item in df['label']]
    merged_data = []  # 创建一个空列表，用于存储合并后的 JSON 数据
    merged_subject = []

    for file, label in zip(json_files, label_list):
        file_path = os.path.join(folder_path, file)  # 构建完整的文件路径
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)  # 读取 JSON 文件并解析为 Python 对象
            conversation = ""
            for index, item in enumerate(data):
                conversation += item['message']
                conversation += "[SEP]"
            conversation_final = conversation[:-5]
            merged_data.append(conversation_final)
            merged_subject.append(file.split('.')[0])
    my_dict = {"merged_subject": merged_subject, "message": merged_data,"label": label_list}
    df_labeled = pd.DataFrame(my_dict)
    df_labeled.to_csv(output_file, index=False)


if __name__ == "__main__":
    folder_path = "./data/task1/trial/subjects"  # 存放 JSON 文件的文件夹路径
    output_file_split = "./data/task1/trial/data_labeled_split.csv"
    output_file_fusion = "./data/task1/trial/data_labeled_fusion.csv"
    label_path = "./data/task1/trial/gold_task1.csv"
    # data_split_csv(label_path, folder_path, output_file_split)
    # data_fusion_csv(label_path, folder_path, output_file_fusion)
