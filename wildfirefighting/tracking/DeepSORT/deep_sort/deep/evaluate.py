#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: evaluate.py
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

import torch

features = torch.load("features.pth")
qf = features["qf"]
ql = features["ql"]
gf = features["gf"]
gl = features["gl"]

# matrix multiplication
scores = qf.mm(gf.t())
res = scores.topk(5, dim = 1)[1][:, 0]
top1correct = gl[res].eq(ql).sum().item()

print("Acc top1: {:.3f}".format(topcorrect/ql.size(0)))


