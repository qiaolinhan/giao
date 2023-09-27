#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: youtube_download.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-09-23
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: This is a tool to download videos from Youtube
#                 It relies on package: pytube
#                 ```pip install pytube```
#
# ------------------------------------------------------------------------------
from pytube import YouTube 

def Download(link):
    youtubeObject = YouTube(link)
    # youtubeObject = youtubeObject.streams.get_highest_resolution()
    # Adjust the resolution of the videos to download
    youtubeObject = youtubeObject.streams.filter(res = "360p")
    try:
        # add the path where to store the downloaded video(s)
        youtubeObject.download("~/Downloads")
    except:
        print("[ERROR]: Download does not work")
    print("[INFO]: Download is completed successfully")

link = input("Please enter the Youtube video URL:")
Download(link)
