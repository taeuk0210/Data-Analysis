import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD

import os
import re
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# load data
# multiome_train = pd.read_parquet("./open-problems-single-cell-perturbations/multiome_train.parquet", engine="pyarrow")
# multiome_obs_meta = pd.read_csv("./open-problems-single-cell-perturbations/multiome_obs_meta.csv")
# multiome_var_meta = pd.read_csv("./open-problems-single-cell-perturbations/multiome_var_meta.csv")

# adata_obs_meta = pd.read_csv("./open-problems-single-cell-perturbations/adata_obs_meta.csv")
# adata_train = pd.read_parquet("./open-problems-single-cell-perturbations/adata_train.parquet", engine="pyarrow")

# dataset
class DE_trainset(Dataset):
    def __init__(self, train):
        super(DE_trainset, self).__init__()
        self.dataset = train
        self.cell2idx = {cell:i for i,cell in enumerate(sorted(self.dataset.cell_type.unique()))}
        self.comp2idx = {comp:i for i,comp in enumerate(sorted(self.dataset.sm_name.unique()))}
        
    def __getitem__(self, index):
        row = self.dataset.iloc[index, :]
        cell = self.cell2idx[row['cell_type']]
        comp = self.comp2idx[row['sm_name']]
        gene_exp = row[5:]
        return torch.tensor([cell, comp], dtype=torch.int64), torch.tensor(gene_exp, dtype=torch.float64)
    
    def __len__(self):
        return self.dataset.shape[0]

class Submit(Dataset):
    def __init__(self, cell2idx, comp2idx):
        super(Submit, self).__init__()
        self.dataset = pd.read_csv("./open-problems-single-cell-perturbations/id_map.csv")
        self.cell2idx = cell2idx
        self.comp2idx = comp2idx
    def __getitem__(self, index):
        row = self.dataset.iloc[index, :]
        cell = self.cell2idx[row['cell_type']]
        comp = self.comp2idx[row['sm_name']]
        return torch.tensor([cell, comp], dtype=torch.int64), torch.tensor(index, dtype=torch.int64)
    def __len__(self):
        return self.dataset.shape[0]
# mymodel
class BaseBlock(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, upsam):
        super(BaseBlock, self).__init__()
        self.layer1 = nn.Linear(dim_in, dim_hidden)
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.layer2 = nn.Linear(dim_hidden, dim_out)
        self.bn2 = nn.BatchNorm1d(dim_out)
        self.relu = nn.ReLU()
        self.upsam = nn.Linear(dim_in, dim_out) if upsam is True else None
        
    def forward(self, x):
        x_ = self.relu(self.bn1(self.layer1(x)))
        x_ = self.bn2(self.layer2(x_))
        if self.upsam is not None:
            x = self.upsam(x) 
        return self.relu(x_ + x)
    
class MLP(nn.Module):
    def __init__(self, dim_emb, dim_model, dim_out):
        super(MLP, self).__init__()
        self.cell_embed = nn.Embedding(6, dim_emb)
        self.comp_embed = nn.Embedding(146, dim_emb)
        self.layer1 = BaseBlock(dim_emb, dim_model, dim_model, True)
        self.upsam1 = BaseBlock(dim_model, dim_model, 2*dim_model, True)
        self.layer2 = BaseBlock(2*dim_model, 2*dim_model, 2*dim_model, False)
        self.upsam2 = BaseBlock(2*dim_model, 2*dim_model, 4*dim_model, True)
        self.layer3 = BaseBlock(4*dim_model, 4*dim_model, 4*dim_model, False)
        self.upsam3 = BaseBlock(4*dim_model, 4*dim_model, 8*dim_model, True)
        self.layer4 = BaseBlock(8*dim_model, 8*dim_model, 8*dim_model, False)
        self.layer5 = nn.Linear(8*dim_model, dim_out)
    def forward(self, x):
        cell_emb = self.cell_embed(x[:,0])
        comp_emb = self.comp_embed(x[:,1])
        x = cell_emb + comp_emb
        x = self.upsam1(self.layer1(x))
        x = self.upsam2(self.layer2(x))
        x = self.upsam3(self.layer3(x))
        x = self.layer5(self.layer4(x))
        return x   
    
class BaseConv(nn.Module):
    def __init__(self, in_feature, mid_feature, out_feature):
        super(BaseConv, self).__init__()
        self.conv1 = nn.Conv1d(in_feature, mid_feature, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(mid_feature)
        self.conv2 = nn.Conv1d(mid_feature, out_feature, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(out_feature)    
        self.relu = nn.ReLU()
    def forward(self, x0):
        x1 = self.relu(self.bn1(self.conv1(x0)))
        x = self.relu(x0 + self.bn2(self.conv2(x1)))
        return x
    
class Down(nn.Module):
    def __init__(self, in_feature):
        super(Down, self).__init__()
        self.conv = nn.Conv1d(in_feature, 2*in_feature, 4, 2, 1)
        self.bn = nn.BatchNorm1d(2*in_feature)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x
    
class FCN(nn.Module):
    def __init__(self, dim_emb, dim_model, dim_out, num_layers):
        super(FCN, self).__init__()
        self.cell_embed = nn.Embedding(6, dim_emb)
        self.comp_embed = nn.Embedding(146, dim_emb)
        self.conv1 = nn.Conv1d(2, dim_model, 3, 1, 1)
        self.layer1 = nn.Sequential(*[BaseConv(dim_model, 2*dim_model, dim_model) for _ in range(num_layers)])
        self.upsam1 = Down(dim_model)
        self.layer2 = nn.Sequential(*[BaseConv(2*dim_model, 4*dim_model, 2*dim_model) for _ in range(num_layers)])
        self.upsam2 = Down(2*dim_model)
        self.layer3 = nn.Sequential(*[BaseConv(4*dim_model, 8*dim_model, 4*dim_model) for _ in range(num_layers)])
        self.upsam3 = Down(4*dim_model)
        self.layer4 = nn.Sequential(*[BaseConv(8*dim_model, 16*dim_model, 8*dim_model) for _ in range(num_layers)])
        self.conv2 = nn.Conv1d(8*dim_model, dim_out, dim_emb//8, dim_emb//8, 0)
    def forward(self, x):
        cell_emb = self.cell_embed(x[:,0]).unsqueeze(1)
        comp_emb = self.comp_embed(x[:,1]).unsqueeze(1)
        x = torch.cat([cell_emb, comp_emb], dim=1)
        x = self.conv1(x)
        x = self.upsam1(self.layer1(x))
        x = self.upsam2(self.layer2(x))
        x = self.upsam3(self.layer3(x))
        x = self.conv2(self.layer4(x))
        return x.squeeze()
        
        
    
# training function
class Average:
    def __init__(self, k_fold):
        self.data = [[] for _ in range(k_fold)]
    def __call__(self, k, epoch, loss, loss_valid, mrrmse, mrrmse_valid):
        self.data[k].append([k, epoch, loss, loss_valid, mrrmse, mrrmse_valid])
        
def MRRMSE(pred, y):
    sum_col = np.sum((pred.detach().cpu().numpy() - \
                     y.detach().cpu().numpy())**2, axis=1) / pred.shape[1] 
    sum_row = np.sum(sum_col**0.5, axis=0) / pred.shape[0]
    return sum_row

def select(model, embed_dim, model_dim, num_layers, optim, learning_rate, criterion, beta, device):
    if model=="mlp":
        model = MLP(embed_dim, model_dim, 18211).to(device)
    if model == "fcn":
        model = FCN(embed_dim, model_dim, 18211, num_layers).to(device)
    if optim == "adam":
        optimizer = Adam(model.parameters(), lr=learning_rate)
    if optim == "sgd":
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    if criterion == "l1":
        criterion = nn.SmoothL1Loss(beta=beta)
    if criterion == "l2":
        criterion = nn.MSELoss()
    return model, optimizer, criterion
    
def train_one_epoch(epoch, model, criterion, alpha, optimizer, device, train_loader, valid_loader, avg, k, verbose):    
    total_loss = 0.
    total_mrrmse = 0.
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        ##
        if (pred[0, :] == pred[1, :]).all():
            print("trivial solution !!!")
        ##
        
        # loss T,Nk cell and others
        if x.shape[0] != x[~(x[:,0] == 0) & ~(x[:,0] == 1)].shape[0]:
            # B, Myeloid cell
            pred_b = pred[(x[:,0] == 0) | (x[:,0] == 1)]
            y_b = y[(x[:,0] == 0) | (x[:,0] == 1)]
            loss_b = criterion(pred_b, y_b)
            # T, Nk cells
            pred_t = pred[~(x[:,0] == 0) & ~(x[:,0] == 1)]
            y_t = y[~(x[:,0] == 0) & ~(x[:,0] == 1)]
            loss_t = criterion(pred_t, y_t)
            # total loss
            loss = alpha * loss_t + (1 - alpha) * loss_b
        else:
            loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mrrmse += MRRMSE(pred, y)

    total_valid = 0.
    total_mrrmse_valid = 0.
    if len(valid_loader)!=1:
        model.eval()
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            total_valid += loss.item()
            total_mrrmse_valid += MRRMSE(pred, y)
    # print loss per epoch
    if verbose:
        print("[Fold : %d] [Epoch : %3d] [Loss : %.5f / %.5f] [MRRMSE : %.3f / %.3f]"%(
            k, epoch, total_loss/len(train_loader), total_valid/len(valid_loader),\
            total_mrrmse/len(train_loader), total_mrrmse_valid/len(valid_loader)))
    avg(k, epoch, total_loss/len(train_loader), total_valid/len(valid_loader),
        total_mrrmse/len(train_loader), total_mrrmse_valid/len(valid_loader))

