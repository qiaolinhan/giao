#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: yolo_detection.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-08-29
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description:  openCV framework
#
# ------------------------------------------------------------------------------

# darknet, tensorflow, opencv
# need to download three files:
# --------------------------------
#   * yolov3.cfg
#   * yolov3.weights 
#   * coco.names

import cv2
import numpy as np

# load the yolo algorithm
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# load the names of objct classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

print(classes)

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# for diffenret classes, use different colors
clors = np.random.uniform(0, 255, size = (len(classes, 3)))

# load an image
img = cv2.imread("roo_ser.jpg")
img = cv2.resize(img, None, fx = 0.4, fy = 0.4)

# original weight and height
height, width, channels = img.shape 
# detect from image
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop = False)

for b in blob:
    for n, img_blob in enumerate(b):
        cv2.imshow(str(n), img_blob)# blobed red, green, blue

net.setInput(blob)
outs = net.forward(output_layers)

# showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2) # 10 pixel
            # rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

number_objects_detected = len(boxes)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
# objects_detected = len(boxes)
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        print(label)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 1, (0, 0, 0), color, 3)

cv2.imshow("Image", img)
cv2.destroyAllWindows()
