#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: DenseResidualBlock.py
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale = 0.2):
        super(ResidualDenseBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity = True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias = True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features = 1 * filters)
        self.b2 = block(in_features = 2 * filters)
        self.b3 = block(in_features = 3 * filters)
        self.b4 = block(in_features = 4 * filters)
        self.b5 = block(in_features = 5 * filters)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]
    
    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x 

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale = 0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_block = nn.Sequential(
                ResidualDenseBlock(filters), ResidualDenseBlock(filters), ResidualDenseBlock(filters), 
        )

class Genrator(nn.Module):
    def __init__(self, channels, filters = 64, num_res_blocks = 16, num_upsample  = 2):
        super(Genrator, self).__init__()


