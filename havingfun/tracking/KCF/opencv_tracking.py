#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: opencv_tracking.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-08-29
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: * delete the ids when the tracked object moved
#
# ------------------------------------------------------------------------------

import cv2
import numpy as np
import math
from object_detection import ObjectDetection

# Initialize obkect detection
od = ObjectDetection()


path = 'data/m300/'
file_name = path/"video.mp4"
captured = cv2.VideoCapture(file_name)

# to show the rame numbers
# initialize count
count = 0
center_points_previous_frame = []
tracking_objects = {}
tracking_id = 0
while True:
    ret, frame = captured.read()
    count += 1
    if not ret:
        break

    # point current frame 
    center_points_current_frame = []

    # detect object on frame
    (class_ids, cores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box
        # take the centre of the bounding box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_current_frame.append((cx, cy))
        print("Frame number", count, " ", x, y, w, h)
    # only compare previous and current frame at the beginning
    if count <= 2: # for the first frame, there is no tracking, hence equals 2
        for pt in center_points_current_frame:
            for pt2 in center_points_current_frame:
                distance = math.phpot(pt2[0] - pt[0], pt2[1] - pt[1])
                if distance < 20:
                    tracking_objects[tracking_id] = pt
                    tracking_id += 1
    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_current_frame_copy = center_points_current_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_current_frame:
                for object_id, pt2 in tracking_objects.items():
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt2[1])
                    # update object position
                    # update the opsition of these id
                    if distance < 20:
                        tracking_objects[object_id] = pt
                        object_exists = True

                        if pt in center_points_current_frame:
                            center_points_current_frame.remove(pt)
                        continue

                # remove the id lost
                if not object_exists:
                    tracking_objects.pop(object_id)
                    
    # add new ids found
    for pt in center_points_current_frame:
        tracking_id
        tracking_object[tracking_id] = pt
        tracking_id += 1

    for tracking_id, pt, tracking_objects.items():
        cv2.circle(frame, pt, 3, (255, 0, 0), -1)
        cv2.recctangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(tracking_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), -2)

    print("======> tracking_objects")
    print(tracking_objects)


    print("======> current frame left point")
    print(center_points_current_frame)

    print("======> previous frame")
    print(center_points_previous_frame)

    cv2.circle(frame, pt, 3, (255, 0, 0), -1)
        

    cv2. imshow("Frame", frame)
    
    # make a copy of the points
    center_points_previous_frame = center_points_current_frame.copy()

    key = cv2.waitkey(0)

    if key == 27:
        break

captured.release()
cv2.destroyAllWindows()
# # showing the image
# cv2.imshow("Frame", frame)
# cv2.waitkey(0)


