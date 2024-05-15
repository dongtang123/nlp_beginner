import pandas as pd
import os
import json
import numpy as np


def csv_json(csv_path, root_path):
    df = pd.read_csv(csv_path)
    df_grouped = df.groupby("round")

    for item in df_grouped:
        subject_list = []
        label_list = []
        for item1, item2 in zip(item[1]['merged_subject'], item[1]['label']):
            subject_list.append(item1)
            label_list.append(item2)
        subject_dict = {item1: item2 for item1, item2 in zip(subject_list, label_list)}
        file_path = os.path.join(root_path, f"round{item[0]}.json")
        df_emissions = pd.read_csv('emissions.csv')
        emissions_name_list = ['duration', 'emissions', 'cpu_energy', 'gpu_energy',
                               'ram_energy', 'energy_consumed', 'cpu_count', 'gpu_count',
                               'cpu_model', 'gpu_model', 'ram_total_size', 'country_iso_code']
        emissions_content_list = [
            df_emissions[item1][0] if type(df_emissions[item1][0]) != np.int64 else int(df_emissions[item1][0]) for
            item1
            in emissions_name_list]
        emissions_dict = {item1: item2 for item1, item2 in zip(emissions_name_list, emissions_content_list)}
        my_dict = {"predictions": subject_dict, "emissions": emissions_dict}
        with open(file_path, 'w') as json_file:
            json.dump(my_dict, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    csv_path = os.path.join('./data/task1/trial/data_labeled_split.csv')
    root_path = os.path.join('./data/rounds_trial')
    csv_json(csv_path, root_path)
