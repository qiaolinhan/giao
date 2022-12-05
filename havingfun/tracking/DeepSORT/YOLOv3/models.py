#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: models.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-12-01
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description:darknet for YOLOv3 
#
# ------------------------------------------------------------------------------
import torch.nn as nn
import torch
import numpy as np

hyperparams = module_defs.pop(0)
output_filters = [int(hyperparams["channels"])]
module_list = nn.ModuleList()
for module_i, module_def in enumerate(module_defs):
    modules = nn.Sequential()

    if module_def["type"] == "convolutional":
        # One of the biggest optimzation of YOLOv3 is adding bn every-layer
        # Conv + batch_normalize + activation
        bn = int(module_def["batch_normalize"])
        filters = int(module_def["filters"])
        kernel_size = int(module_def["size"])
        pad = (kernel_size - 1) // 2
        modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels = output_filters[-1],
                    out_channels = filters,
                    kernel_size = kernel_size,
                    stride = int(module_def["stride"]),
                    padding = pad,
                    bias = not bn,
                    ),
                )
        if bn:
            modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum = 0.9, eps = 1e-5))
        if module_def["activation"] == "leaky":
            modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

    elif module_def["type"] == "maxpool":
        kernel_size = int(module_def["size"])
        stride = int(module_def["stride"])
        if kernel_size == 2 and stride == 1:
            modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
        maxpool = nn.MaxPool2d(kernel_size = kernel_size, stride = stride, padding = int((kernel_size - 1) // 2))
        modules.add_module(f"maxpool_{module_i}", maxpool)

    elif module_def["type"] == "upsample":
        upsample = Upsample(scale_factor=int(module_def["stride"]), mode = "nearest")
        modules.add_module(f"upsample_{module_i}", upsample)

    # concatenate
    elif module_def["type"] == "route":
        layers = [int(x) for x in module_def["layers"].split(",")]
        filters = sum([ouput_filters[1:][i] for i in layers])
        modules.add_module(f"route_{module_i}", EmptyLayer())

    # residual architecture, elementwise add
    elif module_def["type"] == "shortcut":
        filters = output_filters[1:][int(module_def["from"])]
        modules.add_module(f"shortcut_{module_i}", EmptyLayer())

    elif module_def["type"] == "yolo":
        anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
        # extract anchors
        anchors = [int(x) for x in module_def["anchors"].split(",")]
        anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
        anchors = [anchors[i] for i in anchor_idxs]
        num_classes = int(hyperparams["height"])
        # Define detection layer
        yolo_layer = YOLOLayer(anchors, num_classes, img_size)
        modules.add_module(f"yolo_{module_i}", yolo_layer)

    # Register module list and number of output filters
    module_list.append(modules)
    output_filters.append(filters)

return hyperparams, module_list

class Darknet(nn.Module):
    def __init__(self, config_path, img_size = 416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype = np.int32)

    def forward(self, x, targets = None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] =="route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split_size()])
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype = np.int32, count = 5)
            self.header_info = header
            self.seen = header[3]
            weights = np.fromfile(f, dtype = np.float32)

        cutoff = None
        if "darknet53.conv.74" in weight_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i ==cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel() # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

class YOLOLayer(nn.Module):
    # Detection layer
    def __init__(self, anchors, num_classes, img_dim = 416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0
        kkkkk

