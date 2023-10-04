#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: coordinate.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-10-04
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: Straightly output the label bboxes' coordinate 
#
# ------------------------------------------------------------------------------
import cv2
import torch
import glob
import os
import sys
from numpy import asarray
import PIL.Image as Image
import pandas
import csv

# # load the model information
weight_path = '/home/qiao/dev/yolov5/weights/20231002/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', weight_path, force_reload = True)

# load the video which to process
frame_path = '/home/qiao/dev/giao/data/videos/20230926/frames/*.png'
imgs = glob.glob(frame_path)
imgs.sort()
save_path = '/home/qiao/dev/giao/data/videos/20230926/result'

# convert color 
# imgs_bgr = [cv2.imread(f) for f in imgs]
# imgs_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in imgs_bgr ]
# results = [model(f) for f in imgs_rgb]
# results.save(save_path)

for i, f in enumerate(imgs):
    img_bgr = cv2.imread(f)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = Image.fromarray(img_rgb)
    result = model(img_rgb)
    box = result.pandas().xyxy[0]
    print(box)
    box.to_csv(save_path + '/%04d.csv'%i)
    i += 1
