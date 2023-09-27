#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: grad-cam.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-03-21
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: 
#
# ------------------------------------------------------------------------------

import numpy as np
import torch
import torchvision 
import cv2

from matplotlib import pyplot as plt 
from PIL import Image 

# input image  
# ---------------------------------------
# Find the target layer
# e.g.: In the ResNet50, there are multiple CNN blocks from layer1 to layer4.
#       Trying to use the block layer4 and get the last layer from it.  

from torchvision.transforms.transforms import ToPILImage
from torchvision import transforms 
model = torchvision.models.resnet50(pretrained = True)
# model.eval()
# print('======> Print the model:\n', model)

target_layer = model.layer4[-1]
# ----------------------------------------------------

pic = cv2.imread('img_path', 1)
img = pic.copy()
Covert_RGB2GBR


