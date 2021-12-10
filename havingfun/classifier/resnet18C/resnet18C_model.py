#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: resnet34_model.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2021-11-11
#
#   @Email:
#
#   @Description:
#
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

# import commonly used modules
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, 'giao/havingfun/modules')
from inconv import Conv0
from doubleconv import Block


class Resnet18C(nn.Module):
    def __init__(self, img_channels, num_classes):
        super(Resnet18C, self).__init__()
        self.inputconv = Conv0(img_channels, out_channels=64)

        # structure = [2, 2, 2, 2], entire = 2 + 2 * (2+2+2+2) = 18,  transformed form resnet32
        self.layer1 = nn.Sequential(
            Block(in_channels=64, out_channels=64),
            Block(in_channels=64, out_channels=64),
            # Block(in_channels=64, out_channels=64),
        )
        self.layer2 = nn.Sequential(
            Block(in_channels=64, out_channels=128),
            Block(in_channels=128, out_channels=128),
            # Block(in_channels=128, out_channels=128),
            # Block(in_channels=128, out_channels=128),
        )
        self.layer3 = nn.Sequential(
            Block(in_channels=128, out_channels=256),
            Block(in_channels=256, out_channels=256),
            # Block(in_channels=256, out_channels=256),
            # Block(in_channels=256, out_channels=256),
            # Block(in_channels=256, out_channels=256),
            # Block(in_channels=256, out_channels=256),
        )
        self.layer4 = nn.Sequential(
            Block(in_channels=256, out_channels=512),
            Block(in_channels=512, out_channels=512),
            # Block(in_channels=512, out_channels=512),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        input = self.inputconv(x)

        residual1 = self.layer1(input)
        residual2 = self.layer2(residual1)
        residual3 = self.layer3(residual2)
        residual4 = self.layer4(residual3)

        res_out = self.avgpool(residual4)
        ap_out = res_out.reshape(res_out.shape[0], -1)
        output = self.fc(ap_out)
        return output


if __name__ == '__main__':

    # batchsize = 1, channels = 3, inputsize = 255*255
    img = torch.randn((4, 3, 255, 255))
    # model = Resnet34(img_channels=3, num_classes=3)
    model = Resnet18C(img_channels=3, num_classes=3)
    print(model.eval())
    preds = model(img)
    print('input shape:', img.shape)
    print('preds shape:', preds.shape)
