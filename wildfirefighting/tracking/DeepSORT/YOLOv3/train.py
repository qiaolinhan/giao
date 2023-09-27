#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: train.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-11-21
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: pip install terminaltables 
#
# ------------------------------------------------------------------------------

# from models import *
# from utils.logger import *
# from utils.utils import *
# from utils.datasets import *
# from utils.parse_config import *
# from test import evaluate

import warnings
warnings.filterwarnings("ignore")

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path of model")
    parser.add_argument("--data_config", type=str, default="~/dev/giao/data/drone_dataset", help="path of data")
    parser.add_argument("--pretrained_weights", type=str, default="weights/darknet53.conv.74", help="if specified starts from checkpoint")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations")
    parser.add_argument("--compute_mAP", default=False, help="if True, compute mAP every tenth")
    parser.add_argument("--multiscale_training", default=True, help="allow for multiscale training")
    opt = parser.parse_args()
    print(opt)

    # logger = Logger("Logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok = True)
    os.makedirs("ckeckpoints", exist_ok = True)

    # get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config("valid")
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified, we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = DataLoader(
            dataset,
            batch_size = opt.batch_size,
            shuffle = True,
            num_workers = opt.n_cpu,
            pin_memory = True,
            collate_fn = dataset.collate_fn,
            )

    optimizer = optim.Adam(model.parameters())

    metrics = [
            "grid_size",
            "loss",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "cls",
            "cls_acc",
            "recall50",
            "recall75",
            "conf_obj",
            "conf_noobj",
            ]

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batch_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad = False)
            print('imgs', imgs.shape)
            print('targets', targets.shape)
            loss, outputs = model(imgs, targets)
            loss.backwards()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()




