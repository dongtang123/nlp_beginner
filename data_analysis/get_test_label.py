import pandas as pd
import os


def get_label(list_name, path):
    df = pd.read_excel(path)
    for item in list_name:
        for item_name, item_neuroticism, item_extroverted, item_open, item_gravity, item_attractiveness in zip(
                df['name'], df["情绪稳定性"], df["外倾性"], df["创造性"], df["责任心"], df["宜人性"]):
            if item == int(item_name):
                print(item_name,item_neuroticism, item_extroverted, item_open, item_gravity,)


if __name__ == "__main__":
    list_human = [51058117, 51058671, 51058739, 51059684, 51059693, 51059749, 51059800, 51060950, 51060974, 51060979,
                  51061036, 51062229, 51062503, 51071511, 51071582, 51071600, 51072836, 51072874, 51073237, 51075420,
                  51076099, 51078102, 51102226, 51102274]
    label_path = os.path.join('label_csv.xlsx')
    get_label(list_human, label_path)
