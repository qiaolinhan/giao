#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: deepsort_demo.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-10-13
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description:from https://www.bilibili.com/video/BV1Bq4y1N71k?p=3&spm_id_from=pageDriver&vd_source=af71365a49fe7305e3db14d327de14c9 
#
# ------------------------------------------------------------------------------
# In the demo, the aulthor used YOLOv5 for detection
from object_detector import Detector
import imutils  
import cv2

VIDEO_PATH = "./video/test_person.mp4"
RESULT_PATH = "result.mp4"

def main():
    func_status = {}
    func_status['headpose'] = None

    name = 'demo'

    det = Detector()  
    cap = cv2.VideoCapture(VIDEO_PATH)  
    fps = int(cap.get(5))  
    print("======> The video FPS:", fps)
    t = int(1000/fps)

    size = None
    videoWriter = None  
    
    while True:

        # try
        _, im = cap.read()
        if im is None:
            break

        result = det.feedCap(im, func_status)  
        result = result['frame']  
        result = imutils.resize(result, height = 500)

        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                    'm', 'p', '4', 'v'
                    )
            VideoWriter = cv2.VideoWriter(
                    RESULT_PATH, fourcc, fps, (result.shape[1], result.shape[0])
                    )
        videoWriter.write(result)


