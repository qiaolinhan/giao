#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: read_img.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-04-18
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: 
#
# ------------------------------------------------------------------------------


import numpy as np  
import cv2  
# open the gray16 image
gray16_image = cv2.imread('sample_ir.png', cv2.IMREAD_ANYDEPTH)
# convert the gray16 image into a gray8
# If set `IMREAD_ANYDEPTH`, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit. 
gray8_image = np.zeros((120, 160), dtype = np.uint8)
# print(gray8_image)
gray8_image = cv2.normalize(gray16_image, gray8_image, 0, 255, cv2.NORM_MINMAX)
gray8_image = np.uint8(gray8_image)

# color the gray8 image using OpenCV colormaps
inferno_palette = cv2.applyColorMap(gray8_image, cv2.COLORMAP_INFERNO)
jet_palette = cv2.applyColorMap(gray8_image, cv2.COLORMAP_JET)
viridis_palette = cv2.applyColorMap(gray8_image, cv2.COLORMAP_VIRDIS)

# show the different thermal color palettes
cv2.imshow("grayscale", gray8_image)
cv2.imshow("inferno", inferno_palette)
cv2.imshow("jet", jet_palette)
cv2.imshow("viridis", viridis_palette)
cv2.waitKey(0)
