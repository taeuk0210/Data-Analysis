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

from tqdm.notebook import tqdm


###########
# Dataset #
###########
def label_to_transition(label, concat):
    size = label.shape[0]
    out = np.zeros((size**2, size**2))
    for i in range(size):
        for j in range(size):
            if j+1 < size and concat=="horizon":
                out[label[i,j],label[i,j+1]] = 1
            if i+1 < size and concat=="vertical":
                out[label[i,j],label[i+1,j]] = 1
    #out = out + out.T
    return out

def transition_to_label(h, v):
    size = int(h.shape[0]**0.5)
    out = np.zeros((size, size)).astype(int) - 1
    out[0,0] = list(set(np.where(h.sum(axis=1) == 0)[0]) &\
                    set(np.where(v.sum(axis=1) == 0)[0]))[0]
    for i in range(size):
        for j in range(size):
            if i+1 < size:
                out[i+1, j] = v[:, out[i, j]].argmax(axis=0)
            if j+1 < size:
                out[i, j+1] = h[:, out[i, j]].argmax(axis=0)
    return out[::-1, ::-1]
                

class Superimpose(Dataset):
    def __init__(self, cfg, img_path_list, label_list, transform=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transform = transform
        self.img_size = cfg["img_size"]
        self.path = cfg["path"]

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img_path = self.path + img_path[1:]

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image_shuffle = self.transform(image)
            image = image_shuffle.unsqueeze(1).view(3, 4, self.img_size//4, self.img_size)
            image = image.transpose(2,3)
            image = image.reshape(3,16,self.img_size//4,self.img_size//4)
            images = image.transpose(2,3).transpose(0,1)
            
        if self.label_list is not None:
            label_prev = torch.tensor(self.label_list[index], dtype=torch.long) - 1
            label = label_prev.reshape(-1).argsort().reshape(4,4)
            
            h = label_to_transition(label, "horizon")
            v = label_to_transition(label, "vertical")
            label_trans = torch.cat([torch.tensor(h).float(), torch.tensor(v).float()], dim=1)
            return image_shuffle, images, label_trans, label_prev
        else:
            return image_shuffle, images

    def __len__(self):
        return len(self.img_path_list)
    
def Loader(cfg):
    train_path = cfg["path"] + "/train.csv"
    test_path = cfg["path"] + "/test.csv"
    
    df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_transform = transforms.Compose([
        transforms.RandomGrayscale(),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomApply([
            transforms.Grayscale(3),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            ], p=0.25),
        transforms.Resize((cfg['img_size'], cfg['img_size']), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((cfg['img_size'], cfg['img_size']), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    df = df.iloc[np.random.choice(len(df),size=len(df),replace=False)]
    if cfg["train_size"] == 1.:
        train_df = df
        train_labels = train_df.iloc[:,2:].values.reshape(-1, 4, 4)
        
    else:
        train_df = df.iloc[:int(len(df) * cfg["train_size"])]
        valid_df = df.iloc[int(len(df) * cfg["train_size"]):]

        train_labels = train_df.iloc[:,2:].values.reshape(-1, 4, 4)
        valid_labels = valid_df.iloc[:,2:].values.reshape(-1, 4, 4)

    dataset = Superimpose
    
    train_dataset = dataset(cfg, train_df['img_path'].values, train_labels, train_transform)
    train_loader = DataLoader(train_dataset, batch_size = cfg['batch_size'], shuffle=True, num_workers=2)

    test_dataset = dataset(cfg, test_df['img_path'].values, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size = cfg['batch_size'], shuffle=False, num_workers=2)
    
    if cfg["train_size"] == 1.:
        valid_loader = None
    else:
        valid_dataset = dataset(cfg, valid_df['img_path'].values, valid_labels, test_transform)
        valid_loader = DataLoader(valid_dataset, batch_size = cfg['batch_size'], shuffle=False, num_workers=2)
    
    return train_loader, valid_loader, test_loader  


class Superimpose_v2(Dataset):
    def __init__(self, cfg, transform=None, valid=False):
        self.transform = transform
        self.img_size = cfg["img_size"]
        self.path = "/home/aiuser/taeuk/puzzle_data"
        if valid:
            self.images = sorted(os.listdir(self.path))[:2000]
        else:
            self.images = sorted(os.listdir(self.path))

    def __getitem__(self, index):
        img_path = self.path + "/" + self.images[index]
        image = Image.open(img_path)
        image_shuffle, label = shuffle_image(image)
        
        if self.transform is not None:
            image_shuffle = self.transform(image_shuffle)
        image = image_shuffle.unsqueeze(1).view(3, 4, self.img_size//4, self.img_size)
        image = image.transpose(2,3)
        image = image.reshape(3,16,self.img_size//4,self.img_size//4)
        images = image.transpose(2,3).transpose(0,1)
        
        label_prev = label
        label = label_prev.reshape(-1).argsort().reshape(4,4)  
        h = label_to_transition(label, "horizon")
        v = label_to_transition(label, "vertical")
        label_trans = torch.cat([torch.tensor(h).float(), torch.tensor(v).float()], dim=1)
        return image_shuffle, images, label_trans, label_prev
    
    def __len__(self):
        return len(self.images)

def Loader_v2(cfg):
    test_path = "/home/aiuser/kyunghoney/data/Puzzle/test.csv"
    test_df = pd.read_csv(test_path)
    
    train_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomApply([
            transforms.Grayscale(3),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            ], p=0.25),
        transforms.Resize((cfg['img_size'], cfg['img_size']), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((cfg['img_size'], cfg['img_size']), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = Superimpose_v2(cfg, train_transform)
    train_loader = DataLoader(train_dataset, batch_size = cfg['batch_size'], shuffle=True, num_workers=2)

    test_dataset = Superimpose(cfg, test_df['img_path'].values, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size = cfg['batch_size'], shuffle=False, num_workers=2)
    
    valid_dataset = Superimpose_v2(cfg, test_transform, True)
    valid_loader = DataLoader(valid_dataset, batch_size = cfg['batch_size'], shuffle=False, num_workers=2)

    return train_loader, valid_loader, test_loader
    
          






###########
#  model  #
###########


class Residual(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel) 
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ELU()
        self.drop = nn.Dropout2d(0.1)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x))) + x
        return self.drop(x)

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.chnl = [3] + [cfg["channels"]*i*2 for i in range(1, cfg["num_layers"]+1)]
        self.img = cfg["img_size"]//4
        self.pixel = cfg["pixel"]
        self.convs = nn.ModuleList([
            nn.Sequential(
                Residual(self.chnl[i], self.chnl[i+1]),
                nn.MaxPool2d(2,2)
            ) for i in range(len(self.chnl)-1)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(cfg["num_layers"], 2, 1, 1, 0),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            
            nn.Conv2d(2, 2, 1, 1, 0),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            
            nn.Conv2d(2, 2, 1, 1, 0),
        )
        
    def forward(self, x):
        b = x.size(0)
        out = []
        
        for i, conv in enumerate(self.convs):
            x = x.reshape(-1, self.chnl[i],
                          self.img//(2**i), self.img//(2**i))
            x = conv(x)
            x = x.reshape(b, 16, self.chnl[i+1], 
                          self.img//(2**(i+1)), self.img//(2**(i+1)))
            out.append(compare_pixel(x, self.pixel).unsqueeze(1))
        out = torch.cat(out, dim=1)
        out = self.head(out)
        return out
    

        
        
####################################
# Training & Validation _ Customed #
####################################

def train(cfg, device, model, optimizer, scheduler, train_loader, valid_loader=None):
    best_model = None
    best_valid_loss = 100
    title = f'M7_B{cfg["batch_size"]}_L{cfg["num_layers"]}_C{cfg["channels"]}_W{cfg["weight"]}_lr{cfg["learning_rate"]:.3f}_Px{cfg["pixel"]}_'
    
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1-cfg["weight"], cfg["weight"]])).to(device)
    
    for epoch in tqdm(range(1, cfg["num_epochs"]+1)):
        train_loss = []
        valid_loss = []
        train_acc = []
        valid_acc = []
        
        train_preds = []
        train_labels = []
        train_score = 0.
        valid_preds = []
        valid_labels = []
        valid_score = 0.
        
        model.train()
        for _, img, lab, label in iter(train_loader):
            img = img.to(device)
            lab = lab.to(device).long()
            
            optimizer.zero_grad()
            logit = model(img)
            loss = criterion(logit, lab)
            loss.backward()
            optimizer.step()
            
            pred = logit.argmax(dim=1)
            acc = pred.eq(lab).sum() / (lab.size(0)*16*32)
            
            train_loss.append(loss.item())
            train_acc.append(acc.item())
            # check loss #
            if cfg["verbose"]:
                acc_0 = pred.eq(lab)[lab==0].sum() / (lab.size(0)*16*32 - lab.sum())
                acc_1 = pred.eq(lab)[lab==1].sum() / (lab.sum())
                print("Loss : %.6f  Acc : %.4f  0 : %.4f  1 : %.4f"%(
                    loss.item(), acc.item(), acc_0.item(), acc_1.item()))
                
            if cfg["label_prediction"]:
                with torch.no_grad():
                    pred = logit.softmax(dim=1)[:,1,:,:]
                    train_preds.append(pred.detach().cpu())
                    train_labels.append(label.reshape(-1, 16))
     
        if cfg["label_prediction"]:
            hvs = torch.cat(train_preds, dim=0).numpy()
            train_preds = predict_label(hvs).numpy()
            train_labels = torch.cat(train_labels, dim=0).numpy()
            train_score = dacon_measurement(train_labels, train_preds)  

        if valid_loader is not None:
            model.eval()
            with torch.no_grad():
                for _, img, lab, label in valid_loader:
                    img = img.to(device)
                    lab = lab.to(device).long()
    
                    logit = model(img)
                    loss = criterion(logit, lab)
                    
                    pred = logit.argmax(dim=1)
                    acc = pred.eq(lab).sum() / (lab.size(0)*16*32)
                    
                    valid_loss.append(loss.item())
                    valid_acc.append(acc.item())
                    
                    if cfg["label_prediction"]:
                        pred = logit.softmax(dim=1)[:,1,:,:]
                        valid_preds.append(pred.detach().cpu())
                        valid_labels.append(label.reshape(-1, 16))
        else:
            valid_loss, valid_acc = [0], [0]       
        
        if cfg["label_prediction"]:
            hvs = torch.cat(valid_preds, dim=0).numpy()
            valid_preds = predict_label(hvs).numpy()
            valid_labels = torch.cat(valid_labels, dim=0).numpy()
            valid_score = dacon_measurement(valid_labels, valid_preds)  
                
        # scheduler step
        scheduler.step()
        
        # print
        print("[Epoch %3d] [Loss %.7f  %.7f] [Acc %.4f  %.4f] [Score : %.4f  %.4f]"%(
            epoch, np.mean(train_loss), np.mean(valid_loss),
            np.mean(train_acc), np.mean(valid_acc), train_score, valid_score
        ))
        if valid_loss[-1] < best_valid_loss:
            best_valid_loss = valid_loss[-1]
            best_model = model
        
        # save
        if epoch % cfg["save_epoch"] == 0 :
            torch.save(model.state_dict(), cfg["save_path"] + "/" + title + "Ep%d.pt"%epoch)

    return best_model

def dacon_measurement(ground_truth, prediction):
    accuracies = {}
    answer_positions = ground_truth
    submission_positions = prediction
    
    combinations_2x2 = [(i, j) for i in range(3) for j in range(3)]
    combinations_3x3 = [(i, j) for i in range(2) for j in range(2)]
    accuracies['1x1'] = np.mean(answer_positions == submission_positions)

    for size in range(2, 5):
        correct_count = 0
        total_subpuzzles = 0
        for i in range(len(ground_truth)):
            puzzle_a = answer_positions[i].reshape(4, 4)
            puzzle_s = submission_positions[i].reshape(4, 4)
            combinations = combinations_2x2 if size == 2 else combinations_3x3 if size == 3 else [(0, 0)]
            for start_row, start_col in combinations:
                rows = slice(start_row, start_row + size)
                cols = slice(start_col, start_col + size)
                if np.array_equal(puzzle_a[rows, cols], puzzle_s[rows, cols]):
                    correct_count += 1
                total_subpuzzles += 1
                
        accuracies[f'{size}x{size}'] = correct_count / total_subpuzzles
    score = (accuracies['1x1'] + accuracies['2x2'] + accuracies['3x3'] + accuracies['4x4']) / 4.
    return score

def predict_label(hvs):
    result = []
    for i in range(hvs.shape[0]):
        h = hvs[i][:, :16] * (np.ones((16,16)) - np.eye(16))
        v = hvs[i][:, 16:] * (np.ones((16,16)) - np.eye(16))
        answer = solving_v2(h, v)
        answer = answer.argsort()
        result.append(torch.tensor(answer).long().reshape(1,-1))
    return torch.cat(result, dim=0)


#############
## utility ##
#############


def fix_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
def compare_horizon(imgs, i, j, width):
    return torch.sum((imgs[i][:,:,-width:] - imgs[j][:,:,:width])**2)


def compare_vetical(imgs, i, j, width):
    return abs(imgs[i][:,-width:,:] - imgs[j][:,:width,:]).sum().item()


def compare_pixel(imgs, w):
    b = imgs.size(0)
    device = imgs.device
    h = torch.square(imgs[:,:,:,:,-w:].reshape(b, 16, 1, -1) -\
                     imgs[:,:,:,:,:w].reshape(b, 1, 16, -1)).sum(dim=3)
    v = torch.square(imgs[:,:,:,-w:,:].reshape(b, 16, 1, -1) -\
                     imgs[:,:,:,:w,:].reshape(b, 1, 16, -1)).sum(dim=3)

    h = F.normalize(h, p=2, dim=2)
    h = torch.exp(-h / 0.07)
    h = h - (torch.diagonal(h, 0, 1, 2).unsqueeze(2) * \
          torch.eye(16, device=device).expand(h.size(0),16,16)).detach()
    h = h.softmax(dim=1)
    
    v = F.normalize(v, p=2, dim=2)
    v = torch.exp(-v / 0.05)
    v = v - (torch.diagonal(v, 0, 1, 2).unsqueeze(2) * \
          torch.eye(16, device=device).expand(v.size(0),16,16)).detach()
    v = v.softmax(dim=1)

    hv = torch.cat([h,v], dim=2)
    return hv


def solve_one_v1(type, i, available, width, imgs):
    best, p = 1e+10, -1
    
    for j in range(16):
        if available[j]:
            if type == "horizon":
                loss = compare_horizon(imgs, i, j, width)
            elif type == "vertical":
                loss = compare_vetical(imgs, i, j, width)
            else:
                loss = compare_horizon(imgs, i[0], j, width) + \
                    compare_vetical(imgs, i[1], j, width)
            if loss < best:
                p = j
                best = loss
    return p, best


def solving_v1(width, imgs):
    score = 1e+10
    answer = np.zeros((4,4)).astype(int)
    for s in range(16):
        tmp = np.zeros((4,4)).astype(int)
        available = [True] * 16
        tmp[0,0] = s
        available[s] = False
        
        total_loss = 0
        ii = [0,0,1,0,1,2,0,1,2,3,1,2,3,2,3,3]
        jj = [0,1,0,2,1,0,3,2,1,0,3,2,1,3,2,3]
        for i,j in zip(ii, jj):
            if i == 0 and j!=0:
                p, loss = solve_one_v1("horizon", tmp[i,j-1], available, width, imgs)
                tmp[i,j] = p
                available[p] = False
                total_loss += loss
                
            elif j == 0 and i!=0:
                p, loss = solve_one_v1("vertical", tmp[i-1,j], available, width, imgs)
                tmp[i,j] = p
                available[p] = False
                total_loss += loss
                
            elif i !=0 and j !=0:
                p, loss = solve_one_v1("both", [tmp[i,j-1],tmp[i-1,j]] ,available, width, imgs)
                tmp[i,j] = p
                available[p] = False
                total_loss += loss
    
        if score > total_loss:
            score = total_loss
            answer = tmp     
    return answer.reshape(-1)


def solve_one_v2(i, available, h, v=None):
    best_score, p = -1, -1
    for j in range(16):
        if available[j]:
            if v is None:
                score = h[i, j]
            else:
                score = (h[i[0], j] + v[i[1], j])
            if score > best_score:
                p = j
                best_score = score
    if p == -1:
        p = [i for i in range(16) if available[i]][0]
    return p, best_score


def solving_v2(h, v):
    best_score = 0
    answer = np.zeros((4,4)).astype(int)
    for s in range(16):
        tmp = np.zeros((4,4)).astype(int)
        available = [True] * 16
        tmp[0,0] = s
        available[s] = False
        
        total_score = 1
        ii = [0,0,1,0,1,2,0,1,2,3,1,2,3,2,3,3]
        jj = [0,1,0,2,1,0,3,2,1,0,3,2,1,3,2,3]
        
        for i,j in zip(ii, jj):
            if i == 0 and j!=0:
                p, score = solve_one_v2(tmp[i, j-1], available, h)
                tmp[i,j] = p
                available[p] = False
                total_score += score
                
            elif j == 0 and i!=0:
                p, score = solve_one_v2(tmp[i-1, j], available, v)
                tmp[i,j] = p
                available[p] = False
                total_score += score
                
            elif i !=0 and j !=0:
                p, score = solve_one_v2([tmp[i, j-1], tmp[i-1, j]], available, h, v)
                tmp[i,j] = p
                available[p] = False
                total_score += score
        
        if best_score < total_score:
            best_score = total_score
            answer = tmp
            
    return answer.reshape(-1)


def solving_v3(h_prev, v_prev, img):
    
    best_score = 0
    answer = 0
    
    iis = [[0,0,1,0,1,2,0,1,2,3,1,2,3,2,3,3],
           [0,0,1,0,1,2,0,1,2,3,1,2,3,2,3,3],
           [3,3,2,3,2,1,3,2,1,0,2,1,0,1,0,0],
           [3,3,2,3,2,1,3,2,1,0,2,1,0,1,0,0]]
    jjs = [[0,1,0,2,1,0,3,2,1,0,3,2,1,3,2,3],
           [3,2,3,1,2,3,0,1,2,3,0,1,2,0,1,0],
           [0,1,0,2,1,0,3,2,1,0,3,2,1,3,2,3],
           [3,2,3,1,2,3,0,1,2,3,0,1,2,0,1,0]]
    for z in range(4):
        h = h_prev.T if z%2==1 else h_prev
        v = v_prev.T if z//2==1 else v_prev

        for s in range(16):
            tmp = np.zeros((4,4)).astype(int)
            available = [True] * 16
            tmp[0,0] = s
            available[s] = False
            total_score = 0
            
            for i,j in zip(iis[z], jjs[z]):
                if i == 0 and j!=0:
                    p, score = solve_one_v2(tmp[i, j-1], available, h)
                    tmp[i,j] = p
                    available[p] = False
                    total_score += score
                    
                elif j == 0 and i!=0:
                    p, score = solve_one_v2(tmp[i-1, j], available, v)
                    tmp[i,j] = p
                    available[p] = False
                    total_score += score
                    
                elif i !=0 and j !=0:
                    p, score = solve_one_v2([tmp[i, j-1], tmp[i-1, j]], available, h, v)
                    tmp[i,j] = p
                    available[p] = False
                    total_score += score
        
            if best_score < total_score:
                best_score = total_score
                answer = tmp
                
    score = 1e+10
    best = answer
    for id in [[0,1,2,3], [1,0,2,3], [1,2,0,3], [1,2,3,0], [0,2,1,3],
               [0,2,3,1], [2,0,1,3], [0,1,3,2], [3,0,1,2], [0,3,1,2]]:
        for tmp in [answer[id, :], answer[:, id]]:
            tmp_img = shuffle_image(img, tmp.reshape(-1))[0]
            tmp_score = 0.
            for i in range(127, 384, 128):
                tmp_score += np.square(tmp_img[i,:,:]-tmp_img[i+1,:,:]).sum()
                tmp_score += np.square(tmp_img[:,i,:]-tmp_img[:,i+1,:]).sum()
            if tmp_score < score:
                score = tmp_score
                best = tmp

    return best.reshape(-1)


   
    
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


def DeNormalizing(img, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
    img = img.unsqueeze(0) if len(img.shape) == 3 else img
    view_size = (1,3,1,1) if len(img.shape) == 4 else (1,1,3,1,1)
    means = torch.tensor(means).view(*view_size).expand(img.size())
    stds = torch.tensor(stds).view(*view_size).expand(img.size())
    img = img * stds + means
    return img