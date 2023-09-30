#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: frames2video.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-09-29
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: This is the tool to fetch all images in a folder to a video 
#
# ------------------------------------------------------------------------------
import cv2
import numpy as np
import glob
from natsort import natsorted
img_array = []
filenames = natsorted(glob.glob('/home/qiao/dev/giao/data/videos/20230926/UNetruns/*.png'))

# if store the .avi file, it need the fourcc of *'DIVX'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps = 30
frame_size = (640, 480) # width, height
video_name = '/home/qiao/dev/giao/data/videos/20230926/runs/20230926UNet_O.mp4'
for filename in filenames:
    img = cv2.imread(filename)
    # height, width, layers = img.shape
    # size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

