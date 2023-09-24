#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: video.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-09-24
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: 
#
# ------------------------------------------------------------------------------

import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

import numpy as np 
import cv2 
import tensorflow as tf 

if __name__ == "__main__":
    video_path = f"~/dev/giao/havingfun/deving/tools/INSANE Fire Behavior FIRENADO Massive Flames At The Chaparral Fire Murrieta.mp4"

    model = tf.keras.models.load_model("unet.h5")
    model.summary()
