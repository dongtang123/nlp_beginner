import pandas as pd
import os
test_label_path = os.path.join("D:\\data\\nlp_beginner\\classification\\sampleSubmission.csv")

test_data_path = os.path.join("D:\\data\\nlp_beginner\\classification\\test.tsv")
text = pd.read_csv(test_data_path,sep='\t')['Phrase']
label = pd.read_csv(test_label_path)['Sentiment']
merged_df = pd.concat([text, label], axis=1)
merged_file = 'D:\\data\\nlp_beginner\\classification\\merged_data.tsv'
merged_df.to_csv(merged_file, sep='\t', index=False)
