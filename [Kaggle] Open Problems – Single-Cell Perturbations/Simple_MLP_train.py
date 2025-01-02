######################
# training & setting #
######################

import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from Simple_MLP_Regression import Average, DE_trainset, select, train_one_epoch
import torch
from torch.utils.data import DataLoader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
de_train = pd.read_parquet("./open-problems-single-cell-perturbations/de_train.parquet", engine="pyarrow")
shuffle_idx = np.random.choice(de_train.shape[0], de_train.shape[0], replace=False)
de_train = de_train.iloc[shuffle_idx, :]
k_fold = 2

model_name = 'fcn'
num_epochs = 100
batch_size = 64
#embed_dim = 256
#model_dim = 256
num_layers = 4
learning_rate = 0.005
optim="sgd"
criterion = 'l1'
beta = 0.5
alpha = 0.2

for model_dim in tqdm([64,128,256,512]):
    for embed_dim in tqdm([64,128,256,512]):
        avg = Average(k_fold)
        for k in tqdm(range(k_fold)):
        ## k-fold ##
            valid_idx = np.arange(de_train.shape[0])[k*int(de_train.shape[0]/k_fold):(k+1)*int(de_train.shape[0]/k_fold)]
            train_idx = list(set(np.arange(de_train.shape[0])) - set(valid_idx))
            valid_loader = DataLoader(DE_trainset(de_train.loc[valid_idx, :]), batch_size = batch_size, shuffle=True)
            train_loader = DataLoader(DE_trainset(de_train.loc[train_idx, :]), batch_size = batch_size, shuffle=True)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model, optimizer, criterion = select(model_name, embed_dim, model_dim, num_layers, 
                                                 optim, learning_rate, criterion, beta, device)

            for epoch in tqdm(range(1, num_epochs+1)):
                train_one_epoch(epoch, model, criterion, alpha, optimizer, device, train_loader, valid_loader, avg, k, False)
        # save result
        title = f"./SCPsave/{model_name}_batch{batch_size}_epoch{num_epochs}_emb{embed_dim}_model{model_dim}_numL{num_layers}_SGD{learning_rate}_L1_b{beta}_a{alpha}"
        np.save(title + ".npy", np.array(avg.data))
        
        
        ## model train & save ##
        avg = Average(1)
        train_loader = DataLoader(DE_trainset(de_train), batch_size = batch_size, shuffle=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, optimizer, criterion = select(model_name, embed_dim, model_dim, num_layers, 
                                             optim, learning_rate, criterion, beta, device)
        
        for epoch in tqdm(range(1, num_epochs+1)):
            train_one_epoch(epoch, model, criterion, alpha, optimizer, device, train_loader, [1], avg, 0, False)
        
        # save model
        title = f"./SCPsave/{model_name}_batch{batch_size}_epoch{num_epochs}_emb{embed_dim}_model{model_dim}_numL{num_layers}_SGD{learning_rate}_L1_b{beta}_a{alpha}_NF"
        np.save(title + ".npy", np.array(avg.data))
        torch.save({"model":model.state_dict()}, title + ".pt")