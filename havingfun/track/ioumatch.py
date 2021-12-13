import os
import cv2
import numpy as np
from utils import plot_one_box, cal_iou, update_trace_list, draw_trace

# max iou as evidence
initial_target_box = [729, 238, 764, 339]

if __name__ == "__main__":
    video_path = ""
    label_path = ""
    file_name = ""
    last_frame_box = initial_target_box
    cap = cv2.VideoCapter(video_path)
    frame_counter = 1
    cv2.nameWindow("Track", cv2.WINDOW_NORMAL)
    trace_list = []

    SAVE_VIDEO = True
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('maxiououtput.avi', fourcc, 20, (768, 576))

    while(True):
        