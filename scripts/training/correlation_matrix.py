import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os 
from datetime import datetime, timedelta

data_folder = os.path.join(Path(__file__).parent.parent.parent, "data")
filename = os.path.join(data_folder, "dataset_training.csv")
corr_matrix_filename = os.path.join(data_folder, "correlation_matrix_dataset_training.csv")

df = pd.read_csv(filename, encoding='utf-8')

# Sort the df following the chronological order
today = datetime.now().date()
limit = today - timedelta(days=1)

df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime'])
df = df[df['LTScheduledDatetime'].dt.date <= limit]
df = df.sort_values(by='LTScheduledDatetime').reset_index(drop=True)
df = df.select_dtypes(include=['number'])

# splitting depending of the 'Direction' variable:
df_0 = df[df['Direction'] == 0]
df_0 = df_0[["NbPaxTotal","NbPaxTransit","NbPaxConnecting","PxScansCKN","PxScansPIF","PxScansGAT","Direction"]]


df_1 = df[df['Direction'] == 1]



# Calcul de la matrice de corrélation
corr_matrix = df.corr()

print(type(corr_matrix))

corr_matrix.to_csv(corr_matrix_filename, encoding='utf-8')

# with open(corr_matrix_filename, "w") as f:
#     f.write(corr_matrix)
#     f.close()

# # Affichage de la matrice
# print(matrice_correlation)

# # Visualisation avec seaborn
# sns.heatmap(matrice_correlation, annot=False, cmap='coolwarm', center=0)
# plt.title("Matrice de corrélation")
# plt.show()