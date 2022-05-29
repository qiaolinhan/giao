#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: attunet_res34_testing.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-05-17
#
#   @Email: q_linhan@live.concordia.ca
#
#   @Description: This file is to test the attunet_res34 which is for
#                 wildfire segmentation on groundworkstation.
#                 This work is the pre-step for flame points geolocation.
#
# ------------------------------------------------------------------------------

# import some dependencies
import os
import cv2

from fastai import *
from fastai.vision import *
from fastai.vision.all import *

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import pathlib
from datetime import datetime

from tqdm import tqdm

from PIL import Image
from PIL import ImageEnhance
# ----------------------------------
# load the original dataset which is used for building the learner

# the dataset path
path = Path('/home/qiao/dev/giao/datasets/')

# load the images path
path_img = path/'S_kaggle_wildfire/'
fnames = get_image_files(path_img)
# test whether loaded the data correctly
print(f'======> image path: {path_img}')
print(f'======> loaded the images: {fnames[0]}')

# load the label path
path_label = path/'S_kaggle_wildfire_label'
codes = ['Smoke', 'Flame', 'Background']

# ---------------------------------------------
# for building the dataloader

# connection between images and masks names
get_y_fn = lambda x: path_label/f'label_{x.name}'
# building the dataloader for learner
dls = SegmentationDataLoaders.from_label_func(
        path_img,
        bs = 1,
        fnames = fnames,
        label_func = get_y_fn,
        codes = codes,

        # for now, the GPU will out of RAM with image size (960, 770),
        # so resize
        item_tfms = [Resize((400, 400))],
        batch_tfms = [Normalize.from_stats(*imagenet_stats)],
        )

name2id = {v:k for k, v in enumerate(codes)}
void_code = name2id['Background']

# the matrix for learner
def acc_smoke(input, target):
    target = target.squeeze(1)
    mask = target != void_code # mask = target
    return (input.argmax(dim = 1)[mask] == target[mask]).float().mean()
metrics = acc_smoke 

# the model information
learn = unet_learner(dls, models.resnet34, metrics = metrics, 
                    self_attention = True,)
# deploy the learner on device
Device = 'cuda' if torch.cuda.is_available() else 'cpu'
learn.model.to(Device)

# load the trained parameters
learn.load('/home/qiao/dev/giao/datasets/bounding/params/attunet')

# ----------------------------------
# testing part start
# loading test image folder
test_path = '/home/qiao/dev/giao/datasets/bonding/images'
test_fnames = get_image_files(test_path)
total_number = len(test_fnames)
# check whether loaded the testing dataaset correctly
print(f'======> A testing dataset example: {test_fnames[0]}')

# do some convertion just in case it is needed to enhance the original images
# in this work, [brightness, color balance, contrast] are considered, other ratios are also adjust-able.
def enh_img(img):
    # brightness
    enh_bri = ImageEnhance.Brightness(img)
    img_enhanced = enh_bri.enhance(factor = 1.0)

    # ecolor balance
    enh_col = ImageEnhance.Color(img_enhanced)
    img_enhanced = enh_col.enhance(factor = 1.0)

    # contrast
    enh_con = ImageEnhance.Contrast(img_enhanced)
    img_enhanced = enh_con.enhance(factor = 1.0)
    return img_enhanced

def load_image_from_folder(folder_name):
    images = []
    for filename in os.listdir(folder_name):
        if file_name.endwith(".jpg"):
            img = cv2.imread(os.path.join(folder_name, file_name))
            if img is not None:
                images.append(img)
    return images

test_imgs = load_image_from_folder(test_path)
total_number = len(test_imgs)
print(f'======> There are {total_number} images feed.')

for test_img, i in list(test_img):
    # convert cv2 imgs into PIL images
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_PILimg = Image.fromarray(test_img).resize((400, 400))
    
    # predict
    test_predimg = learn.predict(test_PILimg)
    
    # resize and save the predicted masks
    test_outimg = Image.open(test_predimg).resize((960, 770))
    out_imgs.append(test_outimg)
