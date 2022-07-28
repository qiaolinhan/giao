#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: evaluations.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-07-27
#
#   @Email: q_linhan@live.concordia.ca
#
#   @Description: This block stores commonly used evaluation ratios for image
#               classification and segmentation
#
# ------------------------------------------------------------------------------

# nessesary packages
import numpy as np
import sklearn
import sklearn.metrics as metrics

# -----------------------------
# evaluation raios from sklearn
# pixel-level accuracy
def pixelaccuracy(y_true, y_pred):
    pixelaccuracy = metrics.accuracy_score(y_true, y_pred, normalize = False)
    return pixelaccuracy

# ROC-AUC-score: area under the receiver oprating characteristic curve from prediction scores.
def rocaucscore(y_true, y_pred):
    # for the parameter of multi class:
    # orv: one-vs-rest
    # ovo: one-vs-one
    rocaucscore = metrics.roc_auc_score(y_true, y_pred, average = 'macro', multi_class = 'ovr')
    return rocaucscore

# AP, average-precision-score
# summarize a precision-recall curve as the weighted mean of precision achieved at each threshold
def apscore(y_true, y_pred):
    apscore = metrics.average_precision_score(y_true, y_pred, average = 'macro')
    return apscore

# f1-score: balanced f1-score
# F1 = 2 * (precision * recall) / (precision + recall)
def f1score(y_true, y_pred):
    f1score = metrics.f1_score(y_true, y_pred, average = 'macro')
    return f1score
# -----------------------------

