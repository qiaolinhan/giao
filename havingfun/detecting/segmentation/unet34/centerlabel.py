#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: centerlabel.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-06-06
#
#   @Email: q_linhan@live.concordia.ca
#
#   @Description: This file is to note the center of suspect flame or smoke source
#               So that it could be easier to see in the fusioned video and easier
#               for geolocating and its explain.
#
# ------------------------------------------------------------------------------

import argparse
import imutils
import cv2

# constract the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
        help = "path to the input image")
args = vars(ap.parse_args())

# laod the image and convert it into grayscale, 
# blur it slightly (GaussianBlur),
# threshold it 

image = cv2.imread(args["image"])
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# find the edge of these shapes (find contours in the thresholded image)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
    # compute the center of contour
    M = cv2.moments(c)
    cX = int(M["m10"]/M["m00"])
    cY = int(M["m01"]/M["m00"])

    # draw the contour and center of then shape on the image
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(image, "center", (cX-20, cY-20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # show the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
