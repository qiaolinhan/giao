#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: detector.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-10-26
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: for wildfire detection [flame, smoke, background]
#
# ------------------------------------------------------------------------------
import torch
import numpy as np
from models.experimental import attempt_load

from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device


OBJ_LIST = ['flame', 'smoke', 'background']

class Detector(object):
    def __init__(self, weight_path, imgSize = 640, threshold = 0.3, stride = 1):
        super(Detector, self).__init__()
        self.init_model(weight_path)
        self.img_size = imgSize
        self.threshold = threshold 
        self.stride = stride 

    def init_model(self, weight_path):
        self.weights = weight_path 
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location = self.device)
        model.to(self.device).eval()
        model.half()
        self.m = model
        self.names = model.module.names if hasattr(
                model, 'module'
                ) else model.names
        
    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img, new_shape = self.img_size)[0]
        img = img[:, :, ::-1].traspose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img0, img

    def detect(self, im):
        im0, img = self.preprocess(im)
        pred = self.m(img, augment = False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)
        pred_boxes = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape
                        ).round()
                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if not lbl in OBJ_LIST:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                            (x1, y1, x2, y2, lbl, conf)
                            )
        return im, pred_boxes



