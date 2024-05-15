import pandas as pd
import os


if __name__ == "__main__":
    label_path = os.path.join('D:\\nlp_beginner\\data_analysis\\data\\data_1.csv')
    df = pd.read_csv(label_path)
    df.to_excel('label_csv.xlsx',index=False)

