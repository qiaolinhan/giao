#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: mask2boundingbox.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-09-26
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: This is a tool to convert the segmentation mask into bounding box
#
# ------------------------------------------------------------------------------
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mask2border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    # contours = find_contours(mask, 128)
    contours = find_contours(mask, 128)

    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y]  =255

    return border

def mask2bbox(mask):
    bboxes = []
    border = mask2border(mask)
    lbl = label(border)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes

def parse_mask(mask):
    mask = np.expand_dims(mask, axis = -1)
    mask = np.concatenate([mask, mask, mask], axis = -1)
    return mask

if __name__ == "__main__":
    # Load the dataset
    images = glob(os.path.join("bounding", "image", "*"))
    images = sorted(images)
    masks = glob(os.path.join("bounding", "mask", "*"))
    masks = sorted(masks)
    # ir_images = glob(os.path.join("data", "ir", "*"))
    # ir_images = sorted(ir_images)
    # Create folder to save images
    create_dir("runs")
    create_dir("runs/border")
    create_dir("runs/bbox")
    # Loop over the dataset
    for x, y in tqdm(zip(images, masks), total = len(images)):
        print(x)
        # Extract the name
        name = x.split("/")[-1].split(".")[0]
        print("[INFO] The names of data:", name)
        # Read the data and mask
        x = cv2.imread(x, cv2.IMREAD_COLOR) # image
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        # z = cv2.imread(z, cv2.IMREAD_GRAYSCALE)
        # Detecting bounding boxes
        border = mask2border(y) 
        # # save the images
        cv2.imwrite(f"runs/border/{name}.png", border)

        bboxes = mask2bbox(y)
        # masking the bounding box on image
        for bbox in bboxes:
            x = cv2.rectangle(x, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2) 
            # Add the target information
            cv2.putText(x, 'Suspect_Wildfire', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # cat_image = np.concatenate([x, parse_mask(y)], axis = 1)
        cv2.imwrite(f"runs/bbox/{name}.png", x)

