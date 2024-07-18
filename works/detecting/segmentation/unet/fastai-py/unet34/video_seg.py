#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: video_seg.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-09-29
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: Segmentation of the video frames with fastai
#
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from fastai import *
from fastai.vision import *
from fastai.vision.all import * 

import matplotlib.pyplot as plt

print("[INFO] CUDA is available:", torch.cuda.is_available())

path_image = Path('/home/qiao/dev/giao/data/videos/20230926/frames')
codes = ['Smoke', 'Flame_Spot', 'Cloud', 'Person', 'Background']
f_img = get_image_files(path_image)
print("[INFO] Reade the image files\n")

# get_y_fn = lambda x: path_label/f'{x.name}'

# # dataloader
# dls = SegmentationDataLoaders.from_label_func(
#         path_image,
#         bs = 1, # batch_size
#         fnames = f_img,
#         label_func = get_y_fn,
#         codes = codes,
#         item_tfms = [Resize((640, 480))],
#         batch_tfms = [Normalize.from_stats(*imagenet_stats)],
#         )

learn = unet_learner(dls, models.resnet34, metrics = metrics, self_attention = True)
