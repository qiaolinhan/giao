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
#   2023-10-10: Use camera
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
model = torch.hub.load('ultralytics/yolov5', 'custom', weight_path, force_reload = False)

# # load the video which to process
# frame_path = '/home/qiao/dev/giao/data/videos/20230926/frames/*.png'
# imgs = glob.glob(frame_path)
# imgs.sort()


# save_path = '/home/qiao/dev/giao/data/videos/20230926/result'

# for i, f in enumerate(imgs):
#     img_bgr = cv2.imread(f)
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     img_rgb = Image.fromarray(img_rgb)
#     result = model(img_rgb)
#     box = result.pandas().xyxy[0]
#     print(box)
#     box.to_csv(save_path + '/%04d.csv'%i)
#     i += 1

# 20231010
import time
import pathlib
import datetime
get_path = pathlib.Path.cwd()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# params for recording the yolo processed video
fps = 30
frame_size = (640, 480) # width, height
fourcc = cv2.VideoWriter_fourcc(*"MP4V")
date= datetime.datetime.now().strftime("%Y%m%d") 
# print("[INFO]", date)
save_path = '/home/qiao/dev/giao/data/videos/'

record_video_name = save_path + date + 'original.mp4'
result_video_name = save_path + date+ 'yolobboxed.mp4'
# print("[INFO] o name and r name:", record_video_name, result_video_name)
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

original = cv2.VideoWriter(recorde_video_name, fourcc, fps, frame_size)
processed = cv2.VideoWriter(result_video_name, fourcc, fps, frame_size)
while True:
    ret, frame = cam.read()
    original.write(frame)
    cv2.imshow('original', frame)
    print("[INFO] using camera")

    # some preprocessing
    frame_process = frame[:, :, [2, 1, 0]]
    frame_bgr = cv2.imread(frame_process)
    frame_rgb = Image.fromarray(img_rgb)
    result = model(img_rgb)
    processed.write(result)
    cv2.imshow('processed', result)
    print("[INFO] processing the video captured")

    box = result.pandas().xyxy[0]
    print(box)
    box.to_csv(save_path + '/%04d.csv'%i)
    print("[INFO] saving bbox coordinates")

    if cv2.waitkey(1) & 0xFF == ord('q'):
        break

out = cv2.VideoWriter(video_name, fourcc, fps, frame_size)
cam.release()
original.release()
processed.release()
cv2.destroyAllWindows()
