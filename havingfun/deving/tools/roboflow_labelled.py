#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: roboflow_labelled.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-09-12
#
#   @Email: q_linhan@live.concordia.ca
#
#   @Description: 
#
# ------------------------------------------------------------------------------


from roboflow import Roboflow
rf = Roboflow(api_key="yyakrxjBWHEqGKRXtEwM")
project = rf.workspace("concordia-university").project("bounding-for-tracking")
dataset = project.version(2).download("yolov5")

# for terminal
# -----------------
# curl -L "https://app.roboflow.com/ds/bz1BM7KoMZ?key=62dSmL5bzJ" > roboflow.zip;
# unzip roboflow.zip; rm roboflow.zip

