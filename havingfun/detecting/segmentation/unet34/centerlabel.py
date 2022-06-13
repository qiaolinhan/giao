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
import imutils  # pip install --upgrade imutils
import cv2
# constract the argument parse and parse the arguments
print('======> Please input the images path as running *.py -i [images_path]')

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
        help = "path to the input image")
ap.add_argument("-ff", "--contourff",
        help = "to choose bounding flame or entire suspect area")

args = vars(ap.parse_args())
# load the images from folder,
# convert into grayscale, 
# blur slightly (GaussianBlur),
# threshold for distinguish 

# function: load images from folder
def cv_load_folder(folder = args["image"]):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

<<<<<<< HEAD
image = cv2.imread(args["image"])
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)

thresh_flame = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
thresh_smoke = cv2.threshold(blurred, 100, 200, cv2.THRESH_BINARY)[1]
=======
# function: findng the contours in image
def img_contours(image):
    # convert image into grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
>>>>>>> 46c2817fa7f5b56f3bae574dc3f4093af1cc8687

    # blur slightly
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)

    # threshold of flame and smoke, based on our prediction,
    # the value of smoke and flame should be 255/2 and 255,
    # therefore, the threshold for smoke: 100 --> 180,
    # the threshhold for flame: 200 --> 255.
    thresh_flame = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    thresh_smoke = cv2.threshold(blurred, 100, 200, cv2.THRESH_BINARY)[1]
    thresh_wildfire = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]

    # for distinguish smoke and flame:
    if contourff == 0:
        thresh = thresh_smoke
    elif contourff == 1:
        thresh = thresh_wildfire
    else:
        thresh = thresh_flame
    # find the edge of these shapes (find contours in the thresholded image)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

# function: loop over all contours in image and noting the eadges and centers
def bounding_center(cnts):
    # loop over the contours
    for c in cnts:
        # compute the center of contour
        M = cv2.moments(c)
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])

        # draw the contour and center of then shape on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(image, "Suspect wildfire", (cX-20, cY-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # cv2.putText(image, "[疑似火点]", (cX-20, cY-20), 
        #   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # plot and save
        # show the image
        cv2.imshow("Image", cv2.resize(image, (255, 255)))
        cv2.waitKey(0)


if __name__ == "__main__":
    
