##########
# library
##########
import random
import pandas as pd
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm.auto import tqdm
##########
# fix seed 
##########
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
##########
# visualize image and boxes
##########
def draw_boxes_on_image(image_path, annotation_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # txt 파일에서 Class ID와 Bounding Box 정보 읽기
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        values = list(map(float, line.strip().split(' ')))
        class_id = int(values[0])
        x_min, y_min = int(round(values[1])), int(round(values[2]))
        x_max, y_max = int(round(max(values[3], values[5], values[7]))), int(round(max(values[4], values[6], values[8])))

        # 이미지에 바운딩 박스 그리기
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image, str(class_id), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 이미지와 바운딩 박스 출력
    plt.figure(figsize=(25, 25))
    plt.imshow(image)
    plt.show()
##########
# custom collate
##########
def collate_fn(batch):
    images, targets_boxes, targets_labels = tuple(zip(*batch))
    images = torch.stack(images, 0)
    targets = []
    
    for i in range(len(targets_boxes)):
        target = {
            "boxes": targets_boxes[i],
            "labels": targets_labels[i]
        }
        targets.append(target)

    return images, targets
##########
# custom dataset
##########
class CustomDataset(Dataset):
    def __init__(self, root, train=True, transforms=None, pad=None):
        self.root = root
        self.train = train
        self.transforms = transforms
        self.imgs = sorted(glob.glob(root+'/*.png'))
        self.pad = pad
        
        if train:
            self.boxes = sorted(glob.glob(root+'/*.txt'))

    def parse_boxes(self, box_path):
        with open(box_path, 'r') as file:
            lines = file.readlines()

        boxes = []
        labels = []

        for line in lines:
            values = list(map(float, line.strip().split(' ')))
            class_id = int(values[0])
            x_min, y_min = int(round(values[1])), int(round(values[2]))
            x_max, y_max = int(round(max(values[3], values[5], values[7]))), int(round(max(values[4], values[6], values[8])))

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        height, width = img.shape[0], img.shape[1]

        if self.train:
            box_path = self.boxes[idx]
            boxes, labels = self.parse_boxes(box_path)
            # cropping box
            x_min, y_min, x_max, y_max = 2000,2000, 0, 0
            for i in range(boxes.shape[0]):
                x1,y1,x2,y2 = boxes[i]
                x_min = min(x_min, x1, x2)
                x_max = max(x_max, x1, x2)
                y_min = min(y_min, y1, y2)
                y_max = max(y_max, y1, y2)

            if self.pad is not None:
                x_min, x_max, y_min, y_max = x_min-self.pad, x_max+self.pad, y_min-self.pad, y_max+self.pad
            x_center, y_center = (x_min+x_max)/2, (y_min+y_max)/2
            hwh = max(x_max-x_min, y_max-y_min)/2
            x_min, x_max, y_min, y_max = max(0,(x_center-hwh).item()), min(width,(x_center+hwh).item()),\
                                        max(0,(y_center-hwh).item()), min(height,(y_center+hwh).item())
            cropping_box = list(map(int,[x_min, y_min, x_max, y_max]))

            labels += 1 # Background = 0

            if self.transforms is not None:
                transformed = self.transforms(image=img, bboxes=boxes,
                                               labels=labels, augbox=cropping_box)
                img, boxes, labels = transformed["image"], transformed["bboxes"], transformed["labels"]
                
            return img, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

        else:
            if self.transforms is not None:
                transformed = self.transforms(image=img)
                img = transformed["image"]
            file_name = img_path.split('/')[-1]
            return file_name, img, width, height

    def __len__(self):
        return len(self.imgs)
##########
# train transforms 
##########      
def get_train_transforms(img_size,crop_factor):
    return A.Compose([
        A.RandomCropNearBBox((crop_factor,crop_factor),
                              cropping_box_key='augbox'),
        A.Resize(img_size, img_size),
        A.OneOf([
            A.HorizontalFlip(p=0.8),
            A.GaussNoise(p=0.8, var_limit=[0,0.002]),
            A.ColorJitter(p=0.8)
        ]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
##########
# test transforms 
##########
def get_test_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2(),
    ])
##########
# build baseline model
##########
def build_model(num_classes=35):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model
##########
# training function
##########
def train(model, train_loader, optimizer, scheduler, 
          save_interval=10,num_epochs=50, state_path=None,model_name='model',
          device=torch.device('cuda')):
    if state_path is not None:
        load_info = torch.load(state_path)
        start_epoch = load_info['epoch']
        model.load_state_dict(load_info['model'])
        optimizer.load_state_dict(load_info['optimizer'])
        scheduler.load_state_dict(load_info['scheduler'])
    else:
        start_epoch = 0
    
    for epoch in range(start_epoch, start_epoch+num_epochs+1):
        model.train()
        train_loss = []
        for i, (images, targets) in enumerate(tqdm(train_loader)):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()

            loss = model(images, targets)
            # loss 수정
            loss_sum = loss['loss_objectness'] + loss['loss_rpn_box_reg'] + loss['loss_classifier'] + loss['loss_box_reg']
            loss_sum.backward()
            optimizer.step()

            train_loss.append(loss_sum.item())

        if epoch % save_interval==0:
            state = {
                'epoch':epoch+1,
                'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict()}
            torch.save(state, './model_saves/%s_%d'%(model_name, epoch))

        if scheduler is not None:
            scheduler.step()
        
        tr_loss = np.mean(train_loss)

        print(f'Epoch [{epoch}] Train loss : [{tr_loss:.5f}]\n')
        
##########
# bbox denormalizing
##########
def box_denormalize(x1, y1, x2, y2, width, height, img_size):
    x1 = (x1 / img_size) * width
    y1 = (y1 / img_size) * height
    x2 = (x2 / img_size) * width
    y2 = (y2 / img_size) * height
    return x1.item(), y1.item(), x2.item(), y2.item()
##########
# inference function
##########
def inference(model, test_loader, device, title, score_threshold, img_size):
    model.eval()
    model.to(device)
    
    results = pd.read_csv('./sample_submission.csv')

    for img_files, images, img_width, img_height in tqdm(iter(test_loader)):
        images = [img.to(device) for img in images]

        with torch.no_grad():
            outputs = model(images)

        for idx, output in enumerate(outputs):
            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                if score >= score_threshold:
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = box_denormalize(x1, y1, x2, y2, 
                                                     img_width[idx], 
                                                     img_height[idx], img_size)
                    
                    results = results.append({
                        "file_name": img_files[idx],
                        "class_id": label-1,
                        "confidence": score,
                        "point1_x": x1, "point1_y": y1,
                        "point2_x": x2, "point2_y": y1,
                        "point3_x": x2, "point3_y": y2,
                        "point4_x": x1, "point4_y": y2
                    }, ignore_index=True)

    # 결과를 CSV 파일로 저장
    results.to_csv(title, index=False)
    print('Done.')
##########
# draw test image
##########
def draw_image_tset(image_path, df):
    # 이미지 불러오기
    image = cv2.imread('./test/'+image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    lines = df.loc[df.file_name==image_path,:].reset_index()
    for i in range(len(lines)):
        line = lines.iloc[i]
        class_id = int(line.class_id)
        x_min, y_min = int(round(line.point1_x)), int(round(line.point1_y))
        x_max, y_max = int(round(max(line.point2_x,
                                     line.point3_x,
                                     line.point4_x))), \
                        int(round(max(line.point2_y,
                                      line.point3_y,
                                  line.point4_y)))
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image, str(class_id), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 이미지와 바운딩 박스 출력
    plt.figure(figsize=(25, 25))
    plt.imshow(image)
    plt.show()
##########
# classfication dataset
##########
class ClassDataset(Dataset):
    def __init__(self, root, train=True, transforms=None, pad=None, img_size=512):
        self.root = root
        self.train = train
        self.transforms = transforms
        self.imgs = sorted(glob.glob(root+'/*.png'))
        self.pad = pad
        self.boxes = sorted(glob.glob(root+'/*.txt'))
        self.img_size =img_size

    def parse_boxes(self, box_path):
        with open(box_path, 'r') as file:
            lines = file.readlines()
        boxes = []
        labels = []
        for line in lines:
            values = list(map(float, line.strip().split(' ')))
            class_id = int(values[0])
            x_min, y_min = int(round(values[1])), int(round(values[2]))
            x_max, y_max = int(round(max(values[3], values[5], values[7]))), int(round(max(values[4], values[6], values[8])))
            # x1,y1,x2,y2
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)
        return torch.tensor(boxes, dtype=torch.int64), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        height, width = img.shape[0], img.shape[1]
        box_path = self.boxes[idx]
        boxes, labels = self.parse_boxes(box_path)
        # cropping box
        imgs=[]
        for i in range(boxes.shape[0]):
            x_min,y_min,x_max,y_max = boxes[i]

            if self.pad is not None:
                x_min, x_max, y_min, y_max = x_min-self.pad, x_max+self.pad, y_min-self.pad, y_max+self.pad
            x_center, y_center = (x_min+x_max)//2, (y_min+y_max)//2
            hwh = max(x_max-x_min, y_max-y_min)//2
            x_min, x_max, y_min, y_max = max(0,(x_center-hwh).item()), min(width,(x_center+hwh).item()),\
                                        max(0,(y_center-hwh).item()), min(height,(y_center+hwh).item())
            cropping = A.Compose([
                A.Crop(x_min=x_min,y_min=y_min,x_max=x_max,y_max=y_max),
                A.Resize(self.img_size, self.img_size),
                A.OneOf([
                    A.HorizontalFlip(p=0.8),
                    A.GaussNoise(p=0.8, var_limit=[0,0.002]),
                    A.ColorJitter(p=0.8)]),
                ToTensorV2()])
            img_tmp = cropping(image=img)
            imgs.append(img_tmp['image'])
        return torch.stack(imgs,0), labels

    def __len__(self):
        return len(self.imgs)
##########
# classfication dataset collate_fn
##########
def dynamic_collate_fn(batch):
    batch_size = 32
    images, targets = tuple(zip(*batch))
    images = torch.cat(images, dim=0)
    targets = torch.cat(targets, dim=0)
    batch = (images[:batch_size], targets[:batch_size])
    return batch
##########
# classification training function
##########
def train_classification(model, train_loader, optimizer, scheduler, 
          save_interval=10,num_epochs=50, state_path=None,model_name='model',
          device=torch.device('cuda')):
    if state_path is not None:
        load_info = torch.load(state_path)
        start_epoch = load_info['epoch']
        model.load_state_dict(load_info['model'])
        optimizer.load_state_dict(load_info['optimizer'])
        scheduler.load_state_dict(load_info['scheduler'])
    else:
        start_epoch = 0
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, start_epoch+num_epochs+1):
        model.train()
        train_loss = []
        for i, (images, targets) in enumerate(tqdm(train_loader)):

            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()

            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            # escape
            #if i==125:
             #   break
           

        if epoch % save_interval==0:
            state = {
                'epoch':epoch+1,
                'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict()}
            torch.save(state, './model_saves/%s_%d'%(model_name, epoch))

        if scheduler is not None:
            scheduler.step()
        
        tr_loss = np.mean(train_loss)

        print(f'Epoch [{epoch}] Train loss : [{tr_loss:.5f}]\n')