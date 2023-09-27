#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: start_from_0.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-07-29
#
#   @Email: q_linhan@live.concordia.ca
#
#   @Description: 
#
# ------------------------------------------------------------------------------

import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from tqdm import tqdm
num_classes = 2
learning_rate = 1e-5

width = 480
height = 385

batch_size = 4

# create a list of all images in the dataset
# ----------------------------------
from lightdataCV import CVdataset, atransform
# image folder and label folder
TrainFolder = ('datasets/KaggleWildfire20220729/')
ListImages = os.listdir(os.path.join(TrainFolder, 'imgs'))
# some basic transformation
transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)),
    tf.ToTensor(), tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

transformAnn = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)),
    tf.ToTensor()])

# ---------------------------------
# create a function that will allow to load a random image and corresponding annotation map for training
def ReadRandomImage():
    idx = np.random.randint(0, len(ListImages)) # Pick random image

    Img = cv2.imread(os.path.join(TrainFolder, "imgs", ListImages[idx]))
    Mask = cv2.imread(os.path.join(TrainFolder, "labels", "label_"+ListImages[idx]),cv2.IMREAD_GRAYSCALE)
    AnnMap = np.zeros(Img.shape[0:2], np.float32)
    if Mask is not None:
        for n in range(num_classes):
            AnnMap[ Mask == n ] = n/num_classes

    Img = transformImg(Img)
    AnnMap = transformAnn(AnnMap)
    
    return Img, AnnMap

# ----------------------------------
# pick a random index from the list ofiamges and load the iamge corresponding to this index
idx = np.random.randint(0, len(ListImages)) # pick random image
Img = cv2.imread(os.path.join(TrainFolder,
    "imgs", ListImages[idx]))
# load the annotations masks for the image, gray scale
Mask = cv2.imread(os.path.join(TrainFolder, "labels", "label_"+ListImages[idx]), cv2.IMREAD_GRAYSCALE)

# convert the annotation into PyTorch format
AnnMap = np.zeros(Img.shape[0:2], np.float32)
if Mask is not None:
    for n in range(num_classes):
        AnnMap[ Mask == n ] = n/num_classes
Img = transformImg(Img)
AnnMap = transformAnn(AnnMap)

# training
# ---------------------------------
# a batch of image need to be used
# Which means several images stacked on top of each other in a 4D matrix
# Use a function to create the batch
def LoadBatch(): # load batch of images
    images = torch.zeros([batch_size, 3, height, width]) # 3: channels
    ann = torch.zeros([batch_size, height, width]) 

    for i in range(batch_size):
        images[i], ann[i] = ReadRandomImage()
        
    return images, ann

# first part creates [batchsize, channels, height, width]
# load set of images and annotation to the empty matrix, using the ReadRandomImage()
images = torch.zeros([batch_size, 3, height, width]) # 3: channels
ann = torch.zeros([batch_size, height, width]) 

for i in range(batch_size):
    images[i], ann[i] = ReadRandomImage()

# -----------------------------
# load the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from lightunet import LightUnet
Net = LightUnet()
Net = Net.to(device)
optimizer = torch.optim.Adam(params = Net.parameters(), lr = learning_rate)

# -----------------------------
# start the training loop
for itr in range(2000):
    images, ann = LoadBatch()
    images = torch.autograd.Variable(images, requires_grad = False).to(device)
    ann = torch.autograd.Variable(ann, requires_grad = False).to(device)
    pred = Net(images)
