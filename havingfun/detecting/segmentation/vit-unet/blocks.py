#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: blocks.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-09-25
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: Conv bock and Deconv block
#
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size = 3, padding = 1):
        super().__init__() 

        self.layers = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size = kernel_size, padding = padding),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace = True)
                )

    def forward(self, x):
        return self.layer(x)


class DeConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__() 

        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size = 2, stride = 2, padding = 0)

        def forward(self, x):
            return self.deconv(x)


