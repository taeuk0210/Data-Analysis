# library

import random
import pandas as pd
import numpy as np
import os
import glob
import cv2
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm.auto import tqdm
from Baseline_func import *
from model import *

# setting for classfication

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device :', device)
print('Current :', torch.cuda.current_device())
print('Count :', torch.cuda.device_count())

cfg = {
    'img_size':512,
    'hidden_size':48,
    'learning_rate':0.0001,
    'num_epochs':300,
    'batch_size':32,
    'seed':30,
    'save_interval':10
}
seed_everything(cfg['seed'])
# data load
train_dataset = ClassDataset('./train', train=True, transforms=None, pad=70, img_size=cfg['img_size'])
train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, collate_fn=dynamic_collate_fn)

# model, optimizer, scheduler for detector

model = Classifier(img_size=cfg['img_size'], hidden_size=cfg['hidden_size'], num_layers=[4,4,18,4], num_classes = 34)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs'])

# training
# kill process
# kill -9 PID_number

train_classification(model, train_loader, optimizer, scheduler, 
          save_interval=cfg['save_interval'],num_epochs=cfg['num_epochs'], 
          state_path=None,model_name='model1', device=device)