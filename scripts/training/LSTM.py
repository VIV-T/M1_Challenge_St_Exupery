import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from pathlib import Path
import os

# --- 1. CONFIGURATION ---
target_col = 'NbPaxTotal'
window_size = 14
batch_size = 32  # Taille des lots pour le DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Chargement et nettoyage rapide
filename = os.path.join(Path(__file__).parent.parent.parent, "data", 'main_new_preprocessed.csv')
df = pd.read_csv(filename, encoding='utf-8')



# --- 2. PREPARATION DES DONNÉES ---
train_set, test_set = train_test_split(df, test_size=0.10, shuffle=False)

# Sélection numérique et Scaling
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_tr_raw = train_set.select_dtypes(include=['number']).drop(columns=[target_col])
y_tr_raw = train_set[[target_col]]
X_te_raw = test_set.select_dtypes(include=['number']).drop(columns=[target_col])
y_te_raw = test_set[[target_col]]

X_tr_raw = X_tr_raw.loc[:, X_tr_raw.nunique() > 1]
X_te_raw = X_te_raw[X_tr_raw.columns] # Garder les mêmes colonnes pour le test

train_x_sc = scaler_x.fit_transform(X_tr_raw)
train_y_sc = scaler_y.fit_transform(y_tr_raw)
test_x_sc = scaler_x.transform(X_te_raw)
test_y_sc = scaler_y.transform(y_te_raw)

# Fonction Sequence + Création Tenseurs
def create_sequences(data_x, data_y, window):
    X, y = [], []
    for i in range(len(data_x) - window):
        X.append(data_x[i:(i + window)])
        y.append(data_y[i + window])
    return torch.tensor(np.array(X)).float(), torch.tensor(np.array(y)).float()

X_train_seq, y_train_seq = create_sequences(train_x_sc, train_y_sc, window_size)
X_test_seq, y_test_seq = create_sequences(test_x_sc, test_y_sc, window_size)





# --- 3. DATALOADERS ---
# Le DataLoader permet de mélanger les données et de ne pas charger tout le dataset d'un coup
train_dataset = TensorDataset(X_train_seq, y_train_seq)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)







# --- 4. MODÈLE LSTM AVEC BATCH NORMALIZATION ---
class PassengerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PassengerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Batch Normalization 1D (sur les caractéristiques de sortie du LSTM)
        self.bn = nn.BatchNorm1d(hidden_size)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # 'out' est (batch, seq, hidden). On prend le dernier pas de temps : (batch, hidden)
        out = out[:, -1, :]
        
        # Appliquer la Batch Normalization avant la couche linéaire
        out = self.bn(out)
        
        return self.fc(out)

# Initialisation
model = PassengerLSTM(X_tr_raw.shape[1], 64, 2, 1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 5. ENTRAÎNEMENT AVEC DATALOADER ---
epochs = 50
model.train()

print(f"Début de l'entraînement sur {device}...")
for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        # Empeche les gradients d'exploser (limite à 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()
        epoch_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss Moyenne: {epoch_loss/len(train_loader):.6f}")




# --- 6. ÉVALUATION FINALE ---
model.eval()
with torch.no_grad():
    X_test_seq = X_test_seq.to(device)
    preds = model(X_test_seq).cpu().numpy()
    actuals = y_test_seq.numpy()
    
    # Inverse scaling pour interprétation
    final_preds = scaler_y.inverse_transform(preds)
    final_actuals = scaler_y.inverse_transform(actuals)

print(f"\nRMSE final : {np.sqrt(np.mean((final_preds - final_actuals)**2)):.2f}")