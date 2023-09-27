#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: deepsort.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-10-27
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description:this is a conj for deepsort and kf 
#
# ------------------------------------------------------------------------------

from deep_sort.deep_sort import DeepSort # should be a folder named deep_sort

class DeepSortor:
    def __init__(self, configFile):
        cfg = get_config()
        cfg.merge_from_file(configFile)
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist = cfg.DEEPSORT.MAX_DIST, min_confidence = cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap = cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance = cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age = cfg.DEEPSORT.MAX_AGE, 
                            n_init = cfg.DEEPSORT.N_INIT,
                            nn_budget = cfg.DEEPSORT.NN_BUDGET,
                            use_cuda = True)

    def update(self, xywhs, confss, image):
        bboxes2draw = []
        outputs = self.deepsort.update(xywhs, confss, image)

        for value in list(outputs):
            x1, y1, x2, y2, track_id = value
            bboxes2draw.append(
                    (x1, y1, x2, y2, '', track_id)
                    )

        return image, bboxes2draw

