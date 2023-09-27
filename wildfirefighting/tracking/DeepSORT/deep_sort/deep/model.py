#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: model.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-10-27
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description:To extract features of the boxed targets 
#
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

def make_layers(c_in, c_out, repeat_times, is_downsample = False):

class Net(nn.Module):

if __name__ == "__main__":
    net = Net()
    x = torch.randn(4, 3, 128, 64)
    y = net(x)
    print(y.shape)
    import ipdb; ipdb.set_trace()
