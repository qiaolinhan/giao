#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: size_estimation.py
# Author: Linhan Qiao <qiaolinhan073@gmail.com>
# Date: 2024-03-19
# Last Modified By: Linhan Qiao <qiaolinhan073@gmail.com>
# -----
# MIT license

'''
! pip install opencv-contrib-python

To estimate the size of object
Does not matter how far the camera is
Maker gives reference
'''

import cv2
import numpy as np
from ultralytics import YOLO

# Load the aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

# --------------------------
# # Load the img
# img_path = ''
# img = cv2.imread(img_path)

# Load the captured image
cap = cv2.VideoCapture(0)
# # for the camera resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Using while loop to show every frame
while True:
    _, img = cap.read()
    # img = img[0]
    # -------------------------
    # Detect the aruco marker
    corners, _, _ = cv2.aruco.detectMarkers(img, 
                                            aruco_dict,
                                            parameters = parameters)
    # if there is aruco marker, run
    # if no, skip
    if corners:
        # Draw polygon around the marker
        int_corners = np.int0(corners)
        cv2.polylines(img, int_corners, True, (0, 100, 255), 2)

        # Aruco perimeter
        aruco_perimeter = cv2.arcLength(corners[0], True)

        # Pixel to CM ratio
        pixel_cm_ratio = aruco_perimeter/ 20 # 20cm 

        # # Load the object detector
        # from object_detector import *
        # detector = HomogeneousBgDetector()
        model = YOLO("yolov8n-seg.pt")
        # contours = detector.detect_object(img)
        contours = model(img, conf=0.5, boxes=False,
                         show_labels=False,
                         show_conf=False,
                         save_crop=False, save_txt=False,
                         save_conf=False)

        # contours = np.vstack(contours).squeeze()

        # Draw the boundaries on objects
        for cnt in contours:
            print('======>[INFO]:', cnt)

            # Draw poly lines to find the points
            # params:
            #   True -- Closed polygon
            #   (255, 0, 0) -- blue color

            # Acquire the rectangle on polygon
            # Outputs:
            # (x, y) -- center point of the object
            rectangle = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rectangle

            # get (W, h) by applying the ratio pixel to cm
            w_cm = w / pixel_cm_ratio
            h_cm = h / pixel_cm_ratio

            box = cv2.boxPoints(rectangle)
            box = np.int0(box)

            # CAUTION: cv2 can not draw float coordinates 
            # Params:
            #   5 -- how big the circle draws (5 pixels)
            #   (0, 0, 255) -- red color
            #   -1 -- fill the circle with the color
            # draw the center circle
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255)) 
            # draw the polygon
            # cv2.polylines(img, [cnt], True, (255, 0, 0))
            # draw the box
            cv2.polylines(img, [box], True, (255, 0, 0))
            # print the info
            cv2.putText(img, "Width {} cm".format(round(w_cm, 1)), 
                        (int(x-100),int(y-20)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1, (100, 200, 0), 2
                       )
            cv2.putText(img, "Height {} cm".format(round(h_cm, 1)), 
                        (int(x-100),int(y+20)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1, (100, 200, 0), 2
                       )

        cv2.imshow('Img', img)
        key = cv2.waitKey(1)
        if key == 27: # 's' on keyboard
            break

cap.release()
cv2.destroyAllWindows()


# Prepare the aruco marker, 5x5 cm
