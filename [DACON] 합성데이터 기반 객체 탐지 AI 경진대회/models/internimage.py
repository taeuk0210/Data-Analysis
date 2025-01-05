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

# setting for detection

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
    'batch_size':24,
    'seed':30,
    'save_interval':10
}
seed_everything(cfg['seed'])
# data load
train_dataset = CustomDataset('./train', True, get_train_transforms(cfg['img_size'], 0.001), pad=150)
train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_fn)

# build dector model
# https://hyungjobyun.github.io/machinelearning/FasterRCNN2/
def custom_model(img_size, hidden_size, num_layers,
                anchor_sizes=((64,128,256),), anchor_scales=((0.7, 1.0, 1.5),), 
                pooling_out_size=7, pooling_sampling_ratio=2,
                state_path = None):
    
    pretrain = Classifier(img_size, hidden_size, num_layers, num_classes=34)
    state = torch.load(state_path)
    pretrain.load_state_dict(state['model'])
    feature = pretrain.featuremap
    anchor_gen = AnchorGenerator(anchor_sizes, anchor_scales)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=pooling_out_size,
                                                    sampling_ratio=pooling_sampling_ratio)
    box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(in_channels= feature.out_channels*(pooling_out_size**2),
                                                                   representation_size=512) 
    box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(512, num_classes=35)
    model = torchvision.models.detection.FasterRCNN(backbone=feature, num_classes=None, min_size=500, max_size=1000,
                                                    rpn_anchor_generator=anchor_gen, box_predictor=box_predictor,
                                                    box_roi_pool=roi_pooler, box_head=box_head,
                                                    rpn_pre_nms_top_n_train=4000, rpn_pre_nms_top_n_test=4000,
                                                    rpn_post_nms_top_n_train=1300, rpn_post_nms_top_n_test=300,
                                                    rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                                                    rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                                                    box_score_thresh=0.3, box_nms_thresh=0.7, box_detections_per_img=300,
                                                    box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                                                    box_batch_size_per_image=64, box_positive_fraction=0.5)
    return model

# model, optimizer, scheduler for detector

model = custom_model(cfg['img_size'], cfg['hidden_size'], num_layers=[4,4,18,4],
                     anchor_sizes=((100,200,300),), anchor_scales=((0.7, 1.0, 1.5),), 
                     pooling_out_size=7, pooling_sampling_ratio=2, state_path = './model_saves/model1_90')
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs'])


# training
# kill process
# kill -9 PID_number

train(model, train_loader, optimizer, scheduler, 
      save_interval=cfg['save_interval'], num_epochs=cfg['num_epochs'],
      state_path=None, model_name='model1D', device=device)