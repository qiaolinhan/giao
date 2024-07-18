#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: UNet34Seg.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-09-28
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: fastai-based UNet34 for segmentation work, which could be straightly deploied
#
# ------------------------------------------------------------------------------

from re import split
from fastai import *
from fastai.vision import *
from fastai.vision.all import * 

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print('[INFO] The CUDA and nvidia-smi is available:', torch.cuda.is_available())




