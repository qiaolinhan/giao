#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: shell.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-10-27
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: 
#
# ------------------------------------------------------------------------------

from shells import tools
import torch

class Shell(object):
    def __init__(self, deepsort_config_path, yolo_weight_path):
        self.deepsortor = Deepsortor(configFile = deepsort_config_path)
        self.detector = Detector(yolo_weight_path, imgSize = 640, threshould = 0.3, straide = 1)
        self.freamCounter = 0

    def update(self, im):
        # the information needed for output
        retDict = {
                'frame': None,
                'list_of_ids': None,
                'obj_bboxes': []
                }
        self.freamCounter += 1

        # step1: detection on the end of yolov5
        _, bboxes = self.detector.detect(im)
        bbox_xywh = []
        confs = []

        if len(bboxes):
            # Adapt detections to deep sort input format
            for x1, y1, x2, y2, _, conf, in bboxes:
                obj = [
                    int((x1 + x2) / 2), int((y1 + y2) / 2),
                    x2 - x1, y2 - y1
                ]
                bbox_xywh.append(obj)
                confs.append(conf)
            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)

            im, obj_bboxes = self.deepsortor.update(xywhs, confss, im)

            # plotting the results of deepsort
            image = tools.plot_bboxes(im, obj_bboxes)

            retDict['frame'] = image
            retDict['obj_bboxes'] = obj_bboxes

        return retDict
