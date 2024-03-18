#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: homograpy_opencv.py
# Author: Linhan Qiao <qiaolinhan073@gmail.com>
# Date: 2024-03-18
# Last Modified By: Linhan Qiao <qiaolinhan073@gmail.com>
# -----
# MIT license

'''
1. import two images, one is to be registered, anoather
is the reference 
2. convert to gray scales
3. initialte ORB detector
4. find kepoints and describe them
5. match key-points --> Brute force matcher
6. reject bad keypoints which are not able to be
matched --> RANSAC
7. Register two images --> homology
'''
import cv2
import numpy as np

# read the image
img1_path = ''
img1 =  cv2.imread(img1_path)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# initiate ORB
orb = cv2.ORB_create(50)

# key_points and discriptions
kp, des = orb.detectAndCompute(img1, None)

# draw the key points on the original image
img2 = cv2.drawKeypoints(img1, kp, None, flag = None)

cv2.imshow('original', img1)
cv2.imshow('ORB', img2)


# ----------------------------
'''
Create a descriptor matcher of a given type with the
default parameters (using default constructor).
    reval = cv2.DescriptorMatcher_create(cv2.descriptorMatcherType)
    descripterMatcherType:
        * Bruteforce (it uses L2)
        * Bruteforce-L1
        * Bruteforce-Haming
        * Bruteforce-Haming(2)
        * FlannBased
'''

img3_path = ''
img3 = cv2.imread(img3_path)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

kp3, des3 = orb.detectAndCompute(img3, None)

matcher = cv2.DescriptorMatcher_create(cv2.Bruteforce_Haming)
matches = matcher.match(des, des3, None)

# ---------------------------
'''
sort based on the distance transform
'''
# update the matches list by sorting them
matches = sorted(matches, key = lambda x: x.distance)

# draw the matches (draw the top 10 matches)
img4 = cv2.drawMatches(img1, kp1, img3, kp3,
                       matches[:10], None)

cv2.imshow('Matches', img4)


# ---------------------------
# To have good mathces, use homogrophy
# Random sample consensus (RANSAC): eric-yuan.me/ransac/
# In cv, we use RANSAC to calculate homography between
# two images
'''
Needs:
    * All the key_points index

What RANSAC does:
It is nothing but a loop:
    1. Select four feature pairs (at random)
    2. Compute homography H
    3. Compute inliers where ||p_i^', H p_i||<\epsilon
    4. Keep largest set of inlier
    5. Re-compute least-squares H estimate using all of
    the inliers
'''

# for img1
points1 = np.zeros((len(matches), 2), dtype = np.float32)
# for img3
points3 = np.zeros((len(matches), 2), dtype = np.float32)

# To acquire the coordinates of the keypoints mached by
# Bruteforce
for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt
    points3[i, :] = kp3[match.trainIdx].pt

# homography matrix, mask
h, mask = cv2.findHomography(points1, points3, cv2.RANSAC)

# ---------------------------
# Use homography
heght, width, channels = img3.shape()

# using warpPerspective to do image trasformation
'''
cv2.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, bordervalue]]]])
The parameters:
    * src -- input image;
    * dst -- output image that has the size `dsize' and
    the same type as `src';
    * M -- 3x3 transformation matrix;
    * dsize -- size of the output image;
    * flags -- combination of interpolation methods
    (inner_linear or inner_nearest) and the optional flag
    `WARP_INVERSE_MAP', that sets `M' as the inverse
    transformation;
    * borderMode -- pixel extrapolation (BORDER_CONSTANT
    or BORDER_REPLICATE);
    * borderValue -- value used in case of a constant
    border, by default, it equals 0.
'''
img1Reg = cv2.warpPerspective(img1, h, (width, height))

cv2.waitKey(0)
