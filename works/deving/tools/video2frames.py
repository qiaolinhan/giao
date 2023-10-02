#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: Tool_VideoCutter.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-08-28
#
#   @Email: q_linhan@live.concordia.ca
#
#   @Description: To cut video into frame sequence
#
# ------------------------------------------------------------------------------
import cv2
import numpy as np
import os
from pathlib import Path
import glob

path = Path('/home/qiao/dev/giao/data/videos/20230926')
file_posix = path/'20230926.MP4'
file_path = str(file_posix)
# print('[INFO] The file path is:', file_path)

# load the video with cv2
video = cv2.VideoCapture(file_path)

########################################
## play the video
#while video.isOpened():
#    ret, frame = video.read()
#    if not ret:
#        print("[WARNING] Can not receive frame (stream end?).Existing ...")
#        break
#    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
#    cv2.imshow('frame', gray)
#    if cv2.waitKey(1) == ord('q'):
#        break
#video.release()
#cv2.destroyAllWindows()
#######################################

# get the FPS information of the video, also check if we successfully loaded the video
fps = video.get(cv2.CAP_PROP_FPS)
print('[INFO] Fps of the video', fps)
frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
print('[INFO] Frame_count of the video', frame_count)
# calculate the duration, how long the video is, through 'duration = frame_count/fps'
duration = frame_count / fps
print('[INFO] Frame per seconds (FPS) of this video is %d'%fps)
print('[INFO] There are totally %d frames in this video'%frame_count)
print('[INFO] This video remains %d seconds'%duration)

# get the frame ID at a particular time top
hours = 00; minutes = 00; seconds = 60
frame_id = int(fps * (hours ** 60 + minutes * 60 + seconds))
print('[INFO] The specific frame_id at {}:{}:{} is {}'.format(hours, minutes, seconds, frame_id))

################################
## for single frame ############
#video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
#ret, frame = video.read()

## display and save
#cv2.imshow('frame', frame)
#cv2.waitkey(0)
#cv2.imwrite('saved_frame.png', frame)
################################

# cut the video and save into the folder
# for i in range(0, frame_id):
for i in range(0, frame_id):

    video.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = video.read()
    cv2.imwrite('/home/qiao/dev/giao/data/videos/20230926/frames/%04d.png'%i, frame)

print('[INFO] The video is cut into frames and saved in folder')
