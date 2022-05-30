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

# ----------------------------------
# process the images one by one in a folder through 'for loop'

# get the time to create folder for storing predictions
now = datetime.now()
date_time = now.strtime('%Y%m%d%H%M') # Y for year, m for month, d for day, H for hour, M for minute
print(f'======> Current time: {date_time}')

# create the folder
test_pred_path = pathlib.Path(f'/home/qiao/dev/giao/datesets/bounding/pred_{date_time}')
test_pred_path.mkdir(parents = True, exist_ok = True)

# for loop
i = 0
for test_fname in test_fnames:
    # load the data
    test_pilimgimg = Image.open(test_fname)
    test_pilimgimg_resized = test_pilimgimg.resize((255, 255)) # straightly resized here
    test_pilimg = PILImage(test_pilimgimg_resized)

    # enhance inputs
    test_pilimg_enh = emh_img(test_pilimg)
    
    if enhance == True:
        test_pilimg = test_pilimg_enh
    else:
        test_pilimg = test_pilimg

    # predicting by feeding into U-net model
    test_pred_torch_all = learn.predict(test_pilimg)
    test_pred_torch = test_pred_torch_all[0]
    # plt.imshow(test_pred_torch)
    # plt.show()
    
    # to convert predictions: torch.int64 --> PIL.Image
    # convert: torch.int64 --> np.int64
    test_pred_int64 = test_pred_torch[i].numpy() * 255/ 2 # 255/(classes - 1)
    # convert: np.int64 --> np.float64
    test_pred_float64 = np.asarry(test_pred_int64, dtype = np.float64, order = 'C')
    # convert: np.float64 --> PIL.Image
    test_pred_pilimg = Image.fromarray(test_pred_float64)
    # resize to M300 original size (960, 770)
    test_pred_pilimg_resized = test_pred_pilimg.resize((960, 770)).convert('RGB')
    # plt.imshow(test_pred_pilimg_resized)
    # plt.show()

    # fusion the original images and predicted masks for covinient observation
    test_fusion = blend_two_images(test_pilimgimg, test_pred_pilimg_resized)
    # plt.imshow(test_fusion)
    # plt.show()

    # save the results
    # to choose which results it is needed to save
    if fusion_require == True:
        prediction = test_fusion
    else:
        prediction = test_pred_pilimg_resized

    prediction.save(f'{test_pred_path}/premask_{test_fname.name}', 'PNG')

    # countinf as processing
    count = []
    i += 1
    count.append(i)
    print(f'======> prediction of {test_fname.name} saved!')
    print(f'======> It is now counting {count}/{total_number}')


print('\n======> All images are predicted, mission accomplished!!!')
