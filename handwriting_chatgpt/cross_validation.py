import json
import os


def json_info_list(path):
    with open(path, 'r', encoding='utf8') as json_file:
        json_content = json.load(json_file)
    length_0 = 0
    length_1 = 0
    length_2 = 0

    for key, value in json_content.items():
        if value[1]['label'] == 0:
            length_0 += 1
            print(key)
        elif value[1]['label'] == 1:
            length_1 += len(value[0]['text'])
            print(key, len(value[0]['text']))
        elif value[1]['label'] == 2:
            length_2 += len(value[0]['text'])
            print(key, len(value[0]['text']))
    print(length_0, length_1, length_2)


def different(path1, path2):
    with open(path1, 'r', encoding='utf8') as json_file:
        json_content1 = json.load(json_file)
    with open(path2, 'r', encoding='utf8') as json_file:
        json_content2 = json.load(json_file)
        list_key_1 = []
        list_key_2 = []
    for key1, key2 in zip(json_content1.keys(), json_content2.keys()):
        list_key_1.append(key1)
        list_key_2.append(key2)
    list_key_1.sort()
    list_key_2.sort()
    for key1, key2 in zip(list_key_1, list_key_2):
        if len(json_content1[key1][0]['text']) != len(json_content2[key2][0]['text']):
            print(key1,key2)
            print(json_content1[key1][0]['text'],json_content2[key2][0]['text'])


if __name__ == "__main__":
    save_path_1 = os.path.join('aug_data/data_1_test/problem_1_depression_all.json')
    save_path_2 = os.path.join('aug_data/data_1_test/problem_1_anxiety_all.json')
    save_path_3 = os.path.join('aug_data/data_2_test/problem_2_depression_all.json')
    save_path_4 = os.path.join('aug_data/data_2_test/problem_2_anxiety_all.json')
    # json_info_list(save_path_1)
    different(save_path_2, save_path_4)
