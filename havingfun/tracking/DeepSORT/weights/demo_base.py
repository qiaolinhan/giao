#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: demo_base.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-10-26
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: 
#
# ------------------------------------------------------------------------------

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from shells.shell import Shell
import imutils
import cv2

# Video_path
VIDEO_PATH = './video/pedestrain.mp4'
RESULT_PATH = './out/result.mp4'

DEEPSORT_CONFIG_PATH = './deep_sort/configs/deep_sort.yaml'
YOLOV5_WEIGHT_PATH = './weights/yolov5m.py'

def main():
    det = Shell(DEEPSORT_CONFIG_PATH, YOLOV5_WEIGHT_PATH)

    videoWriter = None
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(5))
    t = int(1000/fps) # time
    while True:
        _, frame = cap.read()
        if not _: break

        result = det.update(frame)
        result = result['frame']
        result = imutils.resize(result, height = 500)
        
        # if videoWriter is None:
        #     fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # opencv3.0
        #     videoWriter = cv2.videoWriter(RESULT_PATH, fourcc, fps, (result.shape[1], result.shape[0]))

        cv2.imshow('frame', result)
        key = cv2.waitkey(t)
        if key == ord('q'): break

    cv2.destroyAllWindows() 
    # videoWriter.release()
    cap.release()

if __name__ == '__main__':
    main()

    


