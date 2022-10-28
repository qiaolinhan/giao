#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: ex_convolutionCV.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-10-17
#
#   @Email: q_linhan@live.concordia.ca
#
#   @Description: This is to try https://www.bilibili.com/video/BV12P411V7pc?p=8&spm_id_from=pageDriver&vd_source=af71365a49fe7305e3db14d327de14c9
#
# ------------------------------------------------------------------------------

import cv2
import numpy as np

img = cv2.imread('./datasets/orignial/cats/00000001_00.jpg')

kernel = np.ones((5, 5), np.float32) / 25

dst = cv2.filter2D(img, -1, kernel)

cv2.imshow('img', np.hstack((img,dst)))


cv2.waitKey(0)
cv2.destroyAllWindows()
