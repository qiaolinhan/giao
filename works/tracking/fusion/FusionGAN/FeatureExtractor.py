#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: FeatureExtractor.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-04-21
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: 
#
# ------------------------------------------------------------------------------
import torch.nn as nn
import torch.functional as F
from torchvision.models import vgg19 
import math 
import os 

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained = True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)
