#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: unetr_2d.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-09-25
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: model of unetr 
#
# ------------------------------------------------------------------------------
from blocks import *

import torch
import torch.nn as nn

class UNETR_2D(nn.Module):
    def __init__(self, cf):
        super().__init__()

    def forward(self, x):
        pass 


if __name__ == "__main__":
    config = {}
    config["image_size"] = 256
    config["num_layers"] = 12
    config["hidden_dim"] = 768 
    config["mlp_dim"] = 3072
    config["num_head"] = 12 
    config["dropout_rate"]  =0.1 
    config["num_patches"] = 256 
    config["num_channels"] = 3

    x = torch.randn((
        8,
        config["num_patches"],
        config["pathc_size"] * config["patch_size"] * config["num_channels"],
        ))

