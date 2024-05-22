#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: yolo_seg.py
# Author: Linhan Qiao <qiaolinhan073@gmail.com>
# Date: 2024-03-19
# Last Modified By: Linhan Qiao <qiaolinhan073@gmail.com>
# -----
# MIT license

from ultralytics import YOLO
import cv2

model = YOLO('yolov8n-seg.pt')

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    res = model(img)

    cv2.imshow('Img', res)
    key = cv2.waitKey(1)
    if key == 27: # 's' on keyboard
        break

cap.release()
cv2.destroyAllWindows()
