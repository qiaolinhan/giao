#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: measure_camera_video_tempreture.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-09-18
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: 
#
# ------------------------------------------------------------------------------

import cv2 
import numpy as np 
import os 
import argparse 

# create mouse global coordinates
x_mouse = 0 
y_mouse = 0 

# mouse events function
def mouse_events(event, x, y, flag, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # update mouse global coordinates
        global x_mouse
        global y_mouse 

        x_mouse = x
        y_mouse = y

video_path = "/home/qiao/dev/datasets/videos/20231026/20231002indoor.avi"
# set up the thermal camera index (thermal_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) on Windows OS)
thermal_camera = cv2.VideoCapture(video_path)
# thermal_camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# set up the thermal camera resolution
thermal_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
thermal_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# set up the thermal camera to get the gray16 stream and raw data
thermal_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', '1', '6', ''))
thermal_camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)

# set up mouse events and prepare the thermal frame dispaly
grabbed, frame_thermal = thermal_camera.read()
cv2.imshow('gray8', frame_thermal)
cv2.setMouseCallback('gray8', mouse_events)

# loop over the thermal camera frames
while True:
    # grab the frame from the thermal camera stream
    (grabbed, thermal_frame) = thermal_camera.read()

    # calculate tempreture
    temperature_pointer = thermal_frame[y_mouse, x_mouse]
    temperature_pointer = (temperature_pointer - 32) * 5 / 9

    # convert gray16 to gray8
    cv2.normalize(thermal_frame, thermal_frame, 0, 255,
                  cv2.NORM_MINMAX)
    thermal_frame = np.uint8(thermal_frame)

    # colorized the gray8 image using OpenCV colormaps
    # qiao: need appropriate colormap
    thermal_frame = cv2.applyColorMap(thermal_frame,
                                      cv2.COLORMAP_PLASMA)
    # write pointer
    cv2.circle(thermal_frame, (x_mouse, y_mouse), 2,
               (255, 255, 255), -1)
    # write temperature
    cv2.putText(thermal_frame, "{0:.1f}C".format(temperature_pointer), (x_mouse-40, y_mouse-15),
               cv2.FONT_HERSHEY_PLAIN, 2, (255, 255,255),thickness = 2)

    # to show the frame
    cv2.imshow("gray8", thermal_frame)
    fps=8
    cv2.waitKey(int((1/fps)*1000))
