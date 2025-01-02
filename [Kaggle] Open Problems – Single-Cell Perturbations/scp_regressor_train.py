# library

import os
import re
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD


# load dataset

de_train = pd.read_parquet("/home/aiuser/taeuk/open-problems-single-cell-perturbations/de_train.parquet")
id_map = pd.read_csv("/home/aiuser/taeuk/open-problems-single-cell-perturbations/id_map.csv")
submission = pd.read_csv("/home/aiuser/taeuk/open-problems-single-cell-perturbations/sample_submission.csv")

sm_name_de_train = sorted(de_train.sm_name.unique())
cell_type_de_train = sorted(de_train.cell_type.unique())

# cell type, compound dictionary 

cell_type_dict = {cell_type_de_train[i]:i for i in range(len(cell_type_de_train))}
sm_name_dict = {sm_name_de_train[i]:i for i in range(len(sm_name_de_train))}
print(len(cell_type_dict), len(sm_name_dict))
# mean value

mean_cell_type = de_train.iloc[:, [0]+list(range(5,de_train.shape[1]))].groupby("cell_type").mean().sort_index().reset_index()
mean_sm_name = de_train.iloc[:, [1]+list(range(5,de_train.shape[1]))].groupby("sm_name").mean().sort_index().reset_index()

# SMILES preprocessing

def smile_preprocessing(smile):
    smile = re.sub("[()]", " ", smile)
    smile = re.sub("[1-9]", "1", smile)
    return list(set(smile.split()))

# SMILES decomposition
smiles = de_train[["sm_name", "SMILES"]].drop_duplicates().reset_index(drop=True)
compounds = []
for smile in smiles.SMILES:
    compounds += smile_preprocessing(smile)
compounds = list(set(compounds))
compounds_dict = {compounds[i]:i for i in range(len(compounds))}

smiles = smiles.join(pd.DataFrame(np.zeros((smiles.shape[0], len(compounds)), dtype=np.int32)))
for i in range(smiles.shape[0]):
    coms = list(set(smile_preprocessing(smiles["SMILES"][i])))
    for com in coms:
        smiles.iloc[i, 2 + compounds_dict[com]] = 1
        
compound = smiles.set_index("sm_name").drop("SMILES", axis=1)

# build dataloader

class DEset(Dataset):
    def __init__(self, dataset, cell_type_dict, sm_name_dict, 
                 mean_cell_type, mean_sm_name, compound, gene):
        super(DEset, self).__init__()
        if dataset is not None:
            self.x = dataset.iloc[:, :2]
            self.y = dataset.iloc[:, 5:]
        else:
            self.x = id_map.iloc[:, 1:]
            self.y = None
        self.cell_type_dict = cell_type_dict
        self.sm_name_dict = sm_name_dict
        self.mean_cell_type = mean_cell_type
        self.mean_sm_name = mean_sm_name
        self.compound = compound
        self.gene = gene
        
        
    def __getitem__(self, idx):
        cell, name = self.x.iloc[idx]
        x_cell = self.cell_type_dict[cell]
        x_name = self.sm_name_dict[name]
        x_cell_mean = self.mean_cell_type.iloc[x_cell, self.gene + 1]
        x_name_mean = self.mean_sm_name.iloc[x_name, self.gene + 1]
        #x_compound = np.where(self.compound.loc[name].values==1)[0]
        if self.y is None:
            return torch.tensor([x_cell, x_name], dtype=torch.int64), \
                torch.tensor([x_cell_mean, x_name_mean], dtype=torch.float32)
        else:
            y = self.y.iloc[idx, self.gene]
            return torch.tensor([x_cell, x_name], dtype=torch.int64), \
                    torch.tensor([x_cell_mean, x_name_mean], dtype=torch.float32), \
                    torch.tensor([y], dtype=torch.float32)
                    #torch.tensor(x_compound, dtype=torch.int64), \
                
                
    def __len__(self):
        return self.x.shape[0]
    
# model define
class Basic_Encoder(nn.Module):
    def __init__(self, dim_model):
        super(Basic_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_model, dim_model//2),
            nn.BatchNorm1d(dim_model//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(dim_model//2, dim_model//2),
            nn.BatchNorm1d(dim_model//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(dim_model//2, dim_model//4),
            nn.BatchNorm1d(dim_model//4),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(dim_model//4, dim_model//4),
            nn.BatchNorm1d(dim_model//4),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(dim_model//4, dim_model//8),
            nn.BatchNorm1d(dim_model//8),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, x):
        return self.encoder(x)
    
class Model(nn.Module):
    def __init__(self, dim_model):
        super(Model, self).__init__()
        self.cell_emb = nn.Embedding(6, dim_model)
        self.name_emb = nn.Embedding(146, dim_model)
        self.cell_enc = Basic_Encoder(dim_model)
        self.name_enc = Basic_Encoder(dim_model)
        self.linear = nn.Linear(4, 1)
        
    def forward(self, xs, x_means):
        cell = self.cell_emb(xs[:,0])
        name = self.name_emb(xs[:,1])
        cell_h = self.cell_enc(cell)
        name_h = self.name_enc(name)
        x = torch.cat([cell_h, name_h, x_means], dim=1)
        x = self.linear(x)
        return x
        
# training

def MRRMSE(pred, y):
    pred = pred.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    return np.sqrt(np.square(y - pred).mean(axis=1)).mean()

def train_model(device, train_rate, batch_size, gene, dim_model, learning_rate, num_epochs, verbose):

    
    train_idx = np.random.choice(de_train.shape[0], int(de_train.shape[0]*train_rate), replace=False)
    train_loader = DataLoader(DEset(de_train.iloc[train_idx, :], cell_type_dict,
                                sm_name_dict,mean_cell_type, mean_sm_name, compound, gene),
                            batch_size=batch_size, shuffle=True)
    if train_rate < 1.:
        valid_idx = list(set(np.arange(de_train.shape[0])) - set(train_idx))
        valid_loader = DataLoader(DEset(de_train.iloc[valid_idx, :], cell_type_dict,
                                        sm_name_dict,mean_cell_type, mean_sm_name, compound, gene),
                                batch_size=batch_size, shuffle=True)

    model = Model(dim_model).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss()
    best_mrrmse = 10
    for epoch in range(1, num_epochs+1):
    #for epoch in tqdm(range(1, num_epochs+1)):

        train_loss = 0.
        valid_loss = 0.
        train_mrrmse = 0.
        valid_mrrmse = 0.
        
        model.train()
        for x, m, y in train_loader:
                x, m, y = x.to(device), m.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x,m)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_mrrmse += MRRMSE(pred, y)
        if train_rate < 1.:
                model.eval()
                for x, m, y in valid_loader:
                    x, m, y = x.to(device), m.to(device), y.to(device)
                    with torch.no_grad():
                            pred = model(x,m)
                            loss = criterion(pred, y)
                    
                    valid_loss += loss.item()
                    valid_mrrmse += MRRMSE(pred, y)
                
        # print loss per epoch
        if verbose:
                print("[Epoch : %2d] [Loss : %.4f / %.4f] [MRRMSE : %.3f / %.3f]"%(
                    epoch, train_loss/len(train_loader), valid_loss/len(valid_loader),
                    train_mrrmse/len(train_loader), valid_mrrmse/len(valid_loader)))
        if train_rate < 1. and best_mrrmse > valid_mrrmse/len(valid_loader):
                best_mrrmse = valid_mrrmse/len(valid_loader)
        
    return model, best_mrrmse

def infer_model(device, model, gene):
      test_loader = DataLoader(DEset(None, cell_type_dict, sm_name_dict,
                                      mean_cell_type, mean_sm_name, compound, gene),
                                batch_size=255, shuffle=False)
      model.eval()
      for x, m, in test_loader:
            x, m = x.to(device), m.to(device)
            with torch.no_grad():
                  pred = model(x,m)
      return pred
  
# main

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
train_rate = 1
batch_size = 128
gene = 0
dim_model = 32
learning_rate = 0.025
num_epochs = 100

model, _ = train_model(device, train_rate, batch_size, gene, dim_model, learning_rate, num_epochs, False)
preds = infer_model(device, model, gene)
for gene in range(1, 18211):
    model, _ = train_model(device, train_rate, batch_size, gene, dim_model, learning_rate, num_epochs, False)
    pred = infer_model(device, model, gene)
    preds = torch.cat([preds, pred], dim=1)

data = preds.detach().cpu().numpy()
df = pd.DataFrame(data)
df = df.reset_index(drop=False)
df.columns = submission.columns
df.to_csv("/home/aiuser/taeuk/scp_B%d_D%d_lr%.3f_E%d.csv"%(batch_size, dim_model, learning_rate, num_epochs),
          header=True, index=False)











