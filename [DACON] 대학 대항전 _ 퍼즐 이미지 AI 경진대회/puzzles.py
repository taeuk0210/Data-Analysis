###########
# Library #
###########

import random
import pandas as pd
import numpy as np
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import v2 as transforms
import torchvision

from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore')

###########
# Dataset #
###########

class CustomDataset(Dataset):
    def __init__(self, cfg, img_path_list, label_list, transform=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.path = cfg["path"]
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img_path = self.path + img_path[1:]

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.label_list is not None:
            label = torch.tensor(self.label_list[index], dtype=torch.long) - 1
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)
    
class Superimpose(Dataset):
    def __init__(self, cfg, img_path_list, label_list, transform=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.path = cfg["path"]
        self.img_size = cfg["img_size"]
        self.transform = transform
        self.factors = cfg["factors"]

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img_path = self.path + img_path[1:]

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
            image = image.unsqueeze(1).view(3, 4, self.img_size//4, self.img_size)
            image = image.transpose(2,3)
            image = image.reshape(3,16,self.img_size//4,self.img_size//4)
            image = image.transpose(2,3).transpose(0,1)
            
        if self.label_list is not None:
            images = []
            p = np.random.randint(1,15)
            label = torch.tensor(self.label_list[index], dtype=torch.long) - 1
            label = label.reshape(-1,).argsort().reshape(4,4)            
            
            indices_H = torch.cat([label[:,i:i+2] for i in range(3)], dim=0)
            aug_indices_H = indices_H + torch.cat([torch.zeros(12, 1, dtype=torch.int16),
                                                torch.full((12, 1), p, dtype=torch.int16)], dim=1)
            aug_indices_H %= 16
            indices_H = torch.cat([indices_H, aug_indices_H], dim=0)
                
            indices_V = torch.cat([label[i:i+2,:].transpose(0,1) for i in range(3)], dim=0)
            aug_indices_V = indices_V + torch.cat([torch.zeros(12, 1, dtype=torch.int16),
                                                torch.full((12, 1), p, dtype=torch.int16)], dim=1)
            aug_indices_V %= 16
            indices_V = torch.cat([indices_V, aug_indices_V], dim=0)
                
            images.append(torch.cat([image[indices_H[:,0]], image[indices_H[:,1]]], dim=3))
            images.append(torch.cat([image[indices_V[:,0]], image[indices_V[:,1]]], dim=2).transpose(2,3))
            images = torch.cat(images, dim=0)
            
            labels =  torch.tensor([1]*12 + [0]*12 + [1]*12 + [0]*12,
                                   dtype=torch.float32) 

            return images, labels
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)

class InferDataset(Dataset):
    def __init__(self, cfg, img_path_list, label_list, transform=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.path = cfg["path"]
        self.img_size = cfg["img_size"]
        self.transform = transform
        self.factors = cfg["factors"]

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img_path = self.path + img_path[1:]
        images = []
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
            image = image.unsqueeze(1).view(3, 4, self.img_size//4, self.img_size)
            image = image.transpose(2,3)
            image = image.reshape(3,16,self.img_size//4,self.img_size//4)
            image = image.transpose(2,3).transpose(0,1)
            
            indices = torch.cat([
                torch.arange(16).view(-1,1).expand(16,16).reshape(-1,1),
                torch.arange(16).view(1,-1).expand(16,16).reshape(-1,1)
            ], dim=1)
                
            images.append(torch.cat([image[indices[:,0]], image[indices[:,1]]], dim=3))
            images.append(torch.cat([image[indices[:,0]], image[indices[:,1]]], dim=2).transpose(2,3))
            images = torch.cat(images, dim=0)
            return images

    def __len__(self):
        return len(self.img_path_list)
    
          
def Loader(cfg):
    
    train_path = cfg["path"] + "/train.csv"
    test_path = cfg["path"] + "/test.csv"

    df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    df = df.iloc[np.random.choice(len(df),size=len(df),replace=False)]
    train_df = df.iloc[:int(len(df) * cfg["train_size"])]
    valid_df = df.iloc[int(len(df) * cfg["train_size"]):]

    train_labels = train_df.iloc[:,2:].values.reshape(-1, 4, 4)
    valid_labels = valid_df.iloc[:,2:].values.reshape(-1, 4, 4)

    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
      
    if cfg["superimpose"]:
        dataset = Superimpose
    else:
        dataset = CustomDataset
    if cfg["inference"]:
        dataset = InferDataset
        
    train_dataset = dataset(cfg, train_df['img_path'].values, train_labels, train_transform)
    train_loader = DataLoader(train_dataset, batch_size = cfg['batch_size'], shuffle=True, num_workers=2)

    valid_dataset = dataset(cfg, valid_df['img_path'].values, valid_labels, test_transform)
    valid_loader = DataLoader(valid_dataset, batch_size = cfg['batch_size'], shuffle=False, num_workers=2)

    test_dataset = dataset(cfg, test_df['img_path'].values, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size = cfg['batch_size'], shuffle=False, num_workers=2)
    
    return train_loader, valid_loader, test_loader

# fix random seed 

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
# shuffle

def shuffle_image(img, rand_idx=None, grid=4):
    img = np.array(img)
    shuffled = np.zeros_like(img)
    if rand_idx is None:
        rand_idx = np.random.choice(grid**2, size=grid**2, replace=False)
    for i, idx in enumerate(rand_idx):
        h = (img.shape[0] // grid)
        x, y = (i // grid) * h, (i % grid) * h
        x_, y_ = (idx // grid) * h, (idx % grid) * h
        shuffled[x:x+h, y:y+h,:] = img[x_:x_+h, y_:y_+h,:]
    return shuffled, rand_idx

###########
#  model  #
###########

class MultiHead(nn.Module):
    def __init__(self, cfg):
        super(MultiHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2048, cfg["head_channel"], 1, 1, 0),
            nn.ReLU())
        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(cfg["head_channel"], cfg["head_channel"], 1, 1, 0),
                nn.BatchNorm2d(cfg["head_channel"]),
                nn.LeakyReLU())
            
            for _ in range(cfg["num_layers"])
        ])
        self.classifier = nn.Sequential(
            nn.Conv2d(cfg["head_channel"], cfg["head_channel"]//2, 1, 1, 0),
            nn.BatchNorm2d(cfg["head_channel"]//2),
            nn.ReLU(),
            
            nn.Conv2d(cfg["head_channel"]//2, cfg["head_channel"]//4, 1, 1, 0),
            nn.BatchNorm2d(cfg["head_channel"]//4),
            nn.ReLU(),
            
            nn.Conv2d(cfg["head_channel"]//4, 16, 1, 1, 0),
        )
        
    def forward(self, x):
        output = self.layers(self.conv(x))
        output = self.classifier(output)
        return output

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.backbone = torchvision.models.resnet18(weights="DEFAULT")
        self.out = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(self.out, 1)
    
    def forward(self, x):
        logit = self.backbone(x)
        return logit
    
    
####################################
# Training & Validation _ Customed #
####################################

def train(cfg, model, optimizer, scheduler, train_loader, valid_loader, device):
    best_model = None
    best_valid_loss = 100
    title = f'M2_B{cfg["batch_size"]}_IMG{cfg["img_size"]}_H{cfg["head_channel"]}_lr{cfg["learning_rate"]:.3f}_'
    
    criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    for epoch in tqdm(range(1, cfg["num_epochs"]+1)):
        train_loss = []
        valid_loss = []
        valid_acc = []
        
        model.train()
        for img, lab in iter(train_loader):
            img = img.reshape(-1, 3, cfg["img_size"]//4, cfg["img_size"]//2)
            lab = lab.reshape(-1, 1)
            optimizer.zero_grad()
            logit = model(img.to(device))
            loss = criterion(logit, lab.to(device))
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            
        model.eval()
        with torch.no_grad():
            for img, lab in iter(valid_loader):
                img = img.reshape(-1, 3, cfg["img_size"]//4, cfg["img_size"]//2)
                lab = lab.reshape(-1, 1)
                logit = model(img.to(device))
                loss = criterion(logit, lab.to(device))
                pred = logit.sigmoid()
                pred[pred > 0.5] = 1.
                pred[pred <= 0.5] = 0.
                acc = pred.eq(lab.to(device)).sum() / lab.size(0)
                
                valid_loss.append(loss.item())
                valid_acc.append(acc.item())
                
        # scheduler step
        scheduler.step()
        
        # print
        print(f"[Epoch : {epoch:3d}] [Train : {np.mean(train_loss):.6f}] [Valid : {np.mean(valid_loss):.6f}] [Acc : {np.mean(valid_acc):.3f}]")
        if valid_loss[-1] < best_valid_loss:
            best_valid_loss = valid_loss[-1]
            best_model = model
        
        # save
        if epoch % cfg["save_epoch"] == 0:
            torch.save(model.state_dict(), cfg["save_path"] + "/" + title + "Ep%d.pt"%epoch)    
    return best_model


####################################
# Training & Validation _ Baseline #
####################################

def train_baseline(cfg, model, optimizer, scheduler, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_acc = 0
    best_model = None
    for epoch in range(1, cfg['num_epochs']+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        _val_loss, _val_acc = validation_baseline(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val ACC : [{_val_acc:.5f}]')

        if best_val_acc < _val_acc:
            best_val_acc = _val_acc
            best_model = model
        scheduler.step()        
    return best_model, best_val_acc  


def validation_baseline(cfg, model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    val_acc = []
    with torch.no_grad():
        for imgs, labels in iter(val_loader):
            imgs = imgs.float().to(device)
            imgs = imgs.reshape(-1, 3, cfg["img_size"]//4, cfg["img_size"]//4)
            labels = labels.to(device)
            output = model(imgs)
            loss = criterion(output, labels)
            val_loss.append(loss.item())
            predicted_labels = torch.argmax(output, dim=1)
            for predicted_label, label in zip(predicted_labels, labels):
                val_acc.append(((predicted_label == label).sum() / 16).item())

        _val_loss = np.mean(val_loss)
        _val_acc = np.mean(val_acc)
    return _val_loss, _val_acc

def inference_baseline(cfg, model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            output = model(imgs)
            predicted_labels = torch.argmax(output, dim=1).view(-1, 16)
            predicted_labels = predicted_labels.cpu().detach().numpy()
            preds.extend(predicted_labels)
    submit = pd.read_csv(cfg["path"] + '/sample_submission.csv')
    submit.iloc[:, 1:] = preds
    submit.iloc[:, 1:] += 1  
    return submit