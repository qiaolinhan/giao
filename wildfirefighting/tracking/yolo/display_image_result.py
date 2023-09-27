#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: display_image_result.py
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
import matplotlib
import matplotlib.pyplot as plt

img = cv2.imread('inference/output/street.jpg')
height, width = image.shape[:2]
resized_image = cv2.resize(image, (3* width, 3 * height), interpolation = cv2.INTER_CUBIC)

fig = plt.gcf()
fig.set_size_inches(18, 10)
plt.axis('off')
plt.imshow(cv2.cvtColor(resized_image, cv2.BGR2RGB))
plt.show()


# python detect.py --source inference/videos/street.mp4 --weights yolov5s.pt --conf 0.4
