import json
import os
import matplotlib.pyplot as plt


def read_json_info(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    nums_o = 0
    cate_dict = {}
    valence_dict = {}
    arousal_dict = {}
    for item in json_data:
        nums_o += len(item['Opinion'])
        for cate in item['Category']:
            if cate not in cate_dict:
                cate_dict[cate] = 1
            else:
                cate_dict[cate] += 1
        for intensity in item['Intensity']:
            valence = intensity.split('#')[0]
            arousal = intensity.split('#')[1]
            if valence not in valence_dict:
                valence_dict[valence] = 1
            else:
                valence_dict[valence] += 1
            if arousal not in arousal_dict:
                arousal_dict[arousal] = 1
            else:
                arousal_dict[arousal] += 1
    print(cate_dict)
    print(valence_dict)
    print(arousal_dict)
    print(nums_o)
    return nums_o, cate_dict, valence_dict, arousal_dict


def show_image(data):
    x_values = [float(item[0]) for item in data]
    y_values = [item[1] for item in data]

    # Creating the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x_values, y_values, color='skyblue')
    plt.xlabel('Values', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title('arousal', fontsize=16)
    plt.grid(True)

    plt.show()


def show_image_cate(data):
    categories = list(data.keys())
    counts = list(data.values())
    plt.rcParams['font.sans-serif'] = ['SimSun']
    # Creating the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts, color='skyblue')
    plt.xlabel('Categories', fontsize=16)
    plt.ylabel('Counts', fontsize=16)
    plt.title('Counts of Different Categories', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path_json = os.path.join(
        './dimABSA_TrainingSet&ValidationSet/dimABSA_TrainingSet1/SIGHAN2024_dimABSA_TrainingSet1_Simplified.json')
    nums_opinion, cate_dict, valence_dict, arousal_dict = read_json_info(path_json)
    valence_dict = sorted(valence_dict.items(), key=lambda item: item[0])
    arousal_dict = sorted(arousal_dict.items(), key=lambda item: item[0])
    print(cate_dict)
    print(valence_dict)
    print(arousal_dict)
    # show_image(valence_dict)
    # show_image(arousal_dict)
    show_image_cate(cate_dict)
