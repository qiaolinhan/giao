#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: processvideo.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-02-08
#
#   @Email: q_linhan@live.concordia.ca
#
#   @Description: 
#
# ------------------------------------------------------------------------------

import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

option = {
        'model': 'cfg/yolo.cfg',
        'load': 'bin/yolo.weights',
        'gpu': 1.0
        }

tfnet = TFNet(option)

capture = cv2.VideoCapture('videofile_1080_20fps.avi')
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

# for color in colors:
#     print(color)

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    results = tfnet.return_predict(frame)
    if ret:
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY__COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        print('FPS{:.1f}'.format(1 / (time/time() - stime)))
        if cv2.waitkey(1) & 0xFF == ord('q'):
            break
        else:
            capture.release()
