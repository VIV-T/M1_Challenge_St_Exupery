import pandas as pd



df = pd.read_csv('feature_importance.csv', encoding='utf-8')
print(len(df))

col_to_rmv_list = df['feature'].tail(500).tolist()

print(col_to_rmv_list)


data = pd.read_csv("data/main_preprocessed_new.csv", encoding='utf-8')

data = data.drop(columns=col_to_rmv_list)
print(data.shape)