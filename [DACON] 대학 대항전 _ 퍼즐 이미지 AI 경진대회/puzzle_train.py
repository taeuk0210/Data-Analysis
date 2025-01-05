############
# training #
############

# PID : [1] 1191921
# PID : 

import torch
import torchvision

import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from puzzles_v2 import * 

# num_layers <= 5

cfg = {
    "batch_size":48,
    "img_size":512,
    "train_size":0.999,
    "num_layers":5,
    "channels":8,
    "pixel":2,
    
    "verbose":False,
    "label_prediction":True,
    "weight":0.9,
    "learning_rate":0.002,
    "weight_decay":0.0001,
    "num_epochs":120,
    "save_path":"/home/aiuser/taeuk/puzzle_saves",
    "path":"/home/aiuser/taeuk/puzzle_data",
    "save_epoch":15
}

#fix_random_seed(20240126)
train_loader, valid_loader, test_loader = Loader_v2(cfg)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model(cfg).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"]//5, eta_min=cfg["learning_rate"]/10)

best_model = train(cfg, device, model, optimizer, scheduler, train_loader, valid_loader)

torch.save(best_model.state_dict(), "./best.pt")