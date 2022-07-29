#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: lighttraining.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-07-27
#
#   @Email: q_linhan@live.concordia.ca
#
#   @Description: this .py file is for training the light-weight Unet model, which
#               is build based on ResNet-18 and Attention gate model.
#
# ------------------------------------------------------------------------------

# -----------------------------------------
# import necessary packages
from turtle import clear
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
# for mixed precision training
import torch.cuda.amp as amp 
import sklearn.metrics as metrics
# from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import DataLoader, Subset
# from sklearn.model_selection import train_test_split

from lightdataPIL import JinglingDataset, Atransform
from lightdataCV import CVdataset, atransform 
from lightunet import LightUnet
# from utils import necessary functions
from lightutils import *
# for training performance evaluation
from torchmetrics.detection.mean_ap import MeanAveragePrecision 
from pprint import pprint
import sys
sys.path.insert(1, 'havingfun/deving/blocks')
import evaluations
# Hyperparameters: batch size, number of workers, image size, train_val_split, model
Batch_size = 4
Num_workers = 0
Image_hight = 385
Image_weight = 480
Pin_memory = True
Valid_split = 0.2
Modeluse = LightUnet

# ------------------------------------------
import argparse
# flexible hyper params: epochs, dataset, learning rate, load_model
parser = argparse.ArgumentParser()
# input training epochs
parser.add_argument(
    '-e',
    '--epochs',
    type = int,
    default = 20,
    help = 'Numbers of epochs to train the network'
)
# input learning rate
parser.add_argument(
    '-l',
    '--lr',
    type = np.float32,
    default = 4.04e-7,
    help = 'Learning rate for training'
)
# input the image folder path
parser.add_argument(
    '-t',
    '--troot',
    type = str,
    default = 'datasets/S_kaggle_wildfire',
    help = 'Input the image dataset path'
)
# inputt the label folder path
parser.add_argument(
    '-m',
    '--mroot',
    type = str,
    default = 'datasets/S_kaggle_wildfire_label',
    help = 'Input the mask dataset path'
)

args = vars(parser.parse_args())
Num_epochs = args['epochs']
Learning_rate = args['lr']
Img_dir = args['troot']
Mask_dir = args['mroot']
# ----------------------------------

# ----------------------------------
# the metric to evaluate the model performance of its training
# the device used for training
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print("=====> CUDA is available! Training on GPU...")
else:
    print("=====> CUDA is not available. Training on CPU...")

Device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Device = 'cpu'
# print(f'\nComputation device: {Device}\n')
# ----------------------------------

# ----------------------------------
# load the model
model = Modeluse(in_channels=3, out_channels=1)
model = model.to(Device)
# --------------------------------
# print the parameter numbers of the model
total_params = sum(p.numel() for p in model.parameters())
# print(model.eval())
print('#############################################################')
print(f'======> There are {total_params:,} total parameters in the model.\n')

# ----------------------------------
# load dataset
data = CVdataset(img_dir = Img_dir,mask_dir = Mask_dir, transform = atransform)
dataset_size = len(data)
print(f"Total number of images: {dataset_size}")
# split dataset into training set and validation set
valid_split = 0.2
valid_size = int(valid_split*dataset_size)
indices = torch.randperm(len(data)).tolist()
train_data = Subset(data, indices[:-valid_size])
val_data = Subset(data, indices[-valid_size:])
print(f"======> Total training images: {len(train_data)}")
print(f"======> Total valid_images: {len(val_data)}")


# classes add codes
codes = ['Smoke', 'Flame', 'Background']
name2id = {v:k for k, v in enumerate(codes)}
void_code = name2id['Background']
print('name2id:', name2id)
num_classes = len(name2id)
print('======> num_classes:', num_classes)
# ----------------------------------

# ----------------------------------
# optimizer used for training
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
# loss function for training
loss_fn = nn.CrossEntropyLoss().to(Device)

train_loader = DataLoader(train_data, batch_size = Batch_size, 
                          num_workers = Num_workers, 
                          pin_memory = Pin_memory,
                          shuffle = True)
val_loader = DataLoader(val_data, batch_size = Batch_size, 
                        num_workers = Num_workers, 
                        pin_memory = Pin_memory,
                        shuffle = True)

# ----------------------------------
# functions used in upsampling part
# resize tensor in up-sampling process                        
def sizechange(input_tensor, gate_tensor):
    sizechange = nn.UpsamplingBilinear2d(size = gate_tensor.shape[2:])
    out_tensor = sizechange(input_tensor)
    return out_tensor

# ----------------------------------
# training process
def fit(train_loader, model, optimizer, loss_fn, scaler):
    print('=====> Fitting process')
    loss_p = 0.0
    pacc_p = 0.0
    aucscore_p = 0.0
    f1score_p = 0.0
    apscore_p = 0.0
    counter = 0

    for i, data in tqdm(enumerate(train_loader), total = len(train_data) // Batch_size):
        counter += 1
        img, mask = data
        img = img.to(device = Device)
        mask = mask.to(device = Device)

        # forward
        with amp.autocast(dtype = torch.float16):
            preds = model(img)
            # print('preds size before resize:', preds.size())
            # print('mask size:', mask.size())

            # for multiple class segmentation, the result should be 0, 1, 2, ...
            preds = torch.sigmoid(preds)
            # print('preds size after sigmoid:', preds.size())

            # for now, the predictions are tensors
            # becaus of the U-net characteristic, the output is croped at edges
            # therefore, the tensor need to be resized
            
            # statistics of the loss
            loss = loss_fn(preds, mask)
            loss_p += loss.item()

            preds = preds.squeeze(1).permute(1, 2, 0)
            mask = mask.squeeze(1).permute(1, 2, 0)
            preds = (preds/255).cpu().detach().numpy().astype(np.uint8)
            mask = mask.cpu().detach().numpy().astype(np.uint8)
            print('======> preds size:', preds.shape)
            print('======> masks size:', mask.shape)

            # hist = evaluations.addbatch(preds, mask)
            acc = evaluations.pixelaccuracy(preds, mask)
            pacc_p += acc.item()

            ap = evaluations.preciisonscore(preds, mask)
            ap_p = ap.item()

        # backward
        loss.backward()
        optimizer.step()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() 

        tqdm(enumerate(train_loader)).set_postfix(loss = loss.item(), acc = acc.item(), ap = ap.item())

    epoch_loss = loss_p / counter
    epoch_acc = 100. * pacc_p / counter
    epoch_ap = 100. * ap_p / counter

    # f, ax = plt.subplots(1, 2)
    # ax[0].imshow(preds)
    # ax[1].imshow(mask)
    # plt.show()
    return epoch_loss, epoch_acc, epoch_ap

# def valid(val_loader, model, loss_fn):
#     print('====> Validation process')

#     val_running_loss = 0.0
#     val_running_acc = 0.0
#     val_running_mpa = 0.0
#     counter = 0
#     for i, data in tqdm(enumerate(val_loader), total = len(val_data) // Batch_size):
#         counter += 1

#         img, mask = data
#         img = img.to(device = Device)

#         # mask = mask.unsqueeze(1)
#         # mask = mask.float()
#         mask = mask.to(device = Device)

#         # forward
#         with torch.cuda.amp.autocast():
#             preds = model(img)

#             sig = nn.Sigmoid()
#             preds = sig(preds)
            
#             # if preds.shape != mask.shape:
#             #     # preds = TF.resize(preds, size = mask.shape[2:])
#             #     preds = sizechange(preds, mask)

#             val_loss = loss_fn(preds, mask)
#             val_running_loss += val_loss.item()

#             preds = preds.squeeze(1).permute(1, 2, 0)
#             mask = mask.squeeze(1).permute(1, 2, 0)
#             preds = (preds/255).cpu().detach().numpy().astype(np.uint8)
#             mask = mask.cpu().detach().numpy().astype(np.uint8)

#             # print('preds size:', preds.shape)
#             # print('masks size:', mask.shape)

#             hist = metric.addbatch(preds, mask)
#             val_acc = metric.get_acc()
#             val_running_acc += val_acc.item()
#             val_mpa = metric.get_MPA()
#             val_running_mpa += val_mpa.item()

#         tqdm(enumerate(val_loader)).set_postfix(loss = val_loss.item(), acc = val_acc.item(), mpa = val_mpa.item())

#     val_epoch_loss = val_running_loss / counter
#     val_epoch_acc = 100. * val_running_acc / counter
#     val_epoch_mpa = 100. * val_running_mpa / counter
#     return val_epoch_loss, val_epoch_acc, val_epoch_mpa

def main():

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    # check_accuracy(val_loader, model, device = Device)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(Num_epochs):
        train_epoch_loss, train_epoch_acc, _ = fit(train_loader, model,
                                                    optimizer, loss_fn, scaler)
        # tqdm(enumerate(train_loader)).set_postfix(loss = train_epoch_loss(), acc = train_epoch_loss())
        val_epoch_loss, val_epoch_acc, _ = valid(val_loader, model,loss_fn)
        # tqdm(enumerate(val_loader)).set_postfix(loss = val_epoch_loss.item(), acc = val_epoch_loss())
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        train_acc.append(train_epoch_acc)
        val_acc.append(val_epoch_acc)

        # save entire model
        save_processing_model(Num_epochs, model, optimizer, loss_fn)

        # check accuracy
        # check_accuracy(val_loader, model, device = Device)


        save_training_plots(train_acc, val_acc, train_loss, val_loss)

        # # save final model
    save_entire_model(Num_epochs, model, optimizer, loss_fn)

    print('\n============> TEST PASS!!!\n')


if __name__ == "__main__":
    main()



