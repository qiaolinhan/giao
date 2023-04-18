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
#   @Date: 2023-04-18
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: 
#
# ------------------------------------------------------------------------------

import cv2 
import numpy as np 

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

# set up the thermal camera index (thermal_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) on Windows OS)
thermal_camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# set up the thermal camera resolution
thermal_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
thermal_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

# set up the thermal camera to get the gray16 stream and raw data
thermal_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', '1', '6', ''))
thermal_camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)

# set up mouse events and prepare the thermal frame dispaly
grabbed, frame_thermal = thermal_camera.read()
cv2.imshow('gray8', frame_thermal)
cv2.setMouseCallback('gray8', mouse_event)

# loop over the thermal camera frames
while True:
    # grab the frame from the thermal camera stream
    (grabbed, thermal_frame) = thermal_camera.read()

    # calculate tempreture
    tempreture_pointer = thermal_frame[y_mouse, x_mouse]

