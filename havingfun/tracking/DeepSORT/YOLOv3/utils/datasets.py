#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: datasets.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-11-21
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description:to load the datas 
#
# ------------------------------------------------------------------------------
from torchvision import transforms
from PIL import Image
#############################

def __getitem__(self, index):
    # image
    img_path = self.img_files[index % len(self.img_files)].rstrip()
    img_path = "~/dev/giao/data/drone_dataset" + img_path
    # print(img_path)

    # Extract images as Pytorch tensor
    img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

    # Handle images with less than three channels
    if len(img.shape) != 3:
        img = img.unsqueeze(0)
        img = img.expand((3, img.shape[1:]))

    _, h, w = img.shape
    h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
