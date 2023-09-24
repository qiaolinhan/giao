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
    # # To see the INFO of the model
    # model.summary() 

    # load the video though cv2
    vs = cv2.VideoCapture(video_path) 
    _, frame = vs.read() 
    H, W, _ = frame.shape 
    vs.release() 

    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G") 
    out = cv2.VideoWriter("output.avi", fourcc, 10, (W, H), True) 

    cap = cv2.VideoCapture(video_path) 
    while True:
        ret, frame = cap.read() 

        if ret == False:
            cap.release() 
            out.release() 

        ori_frame = frame
        # --> [256, 256, 3]
        frame = cv2.resize(frame, (256, 256))
        # [256, 256, 3] --> [1, 256, 256, 3]
        frame = np.expand_dims(frame, axis = 0)
        frame = frame/255.0 

        mask = model.predict(frame)
        mask = mask[0]
        mask = mask > 0.5 
        mask = mask.astype(np.float32)
        mask = cv2.resize(mask, (W, H))
        mask = np.expand_dims(mask, axis = -1)
        # print(mask.shape)

        combine_frame = ori_frame * mask 
        combine_frame = combine_frame.astype(np.uint8)

        cv2.imwrite(f"video/{idx}.png", combine_frame)
        idx += 1


