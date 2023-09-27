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
import os
import numpy as np
import torch
#############################

def __getitem__(self, index):
    # ------
    # image
    # ------
    img_path = self.img_files[index % len(self.img_files)].rstrip()
    img_path = "~/dev/giao/data/drone_dataset/train/images" + img_path
    print(img_path)

    # Extract images as Pytorch tensor
    img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

    # Handle images with less than three channels
    if len(img.shape) != 3:
        img = img.unsqueeze(0)
        img = img.expand((3, img.shape[1:]))

    _, h, w = img.shape
    h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

    # padding into square resolution
    img, pad = pad_to_sqare(img, 0)
    _, padded_h, padded_w = img.shappe # 640*640

    # ------
    # Label
    # ------
    label_path = self.label_files[index % len(self.img_files)].rstrip()
    label_path = "~/dev/giao/data/drone_dataset/train/labels" + label_path

    targets = None
    if os.path.exists(label_path):
        boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
        # Extract coordinates for unpadded + unscaled image
        x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)

        # adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]

        # returns (x, y, w, h)
        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        boxes[:, 3] *= w_factor / padded_w
        boxes[:, 4] *= h_factor / padded_h

        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes

    # apply augmentations, largerdataset
    if self.augment:
        if np.random.random() < 0.5:
            img, targets = horisontal_flip(img, targets)

    return img_path, img, targets


