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
        self.cf = cf
        # Path + Position Embeddings
        self.patch_embed = nn.Linear(
            cf["patch_size"] * cf["patch_size"] * cf["num_channels"],
            cf["hidden_dim"]
                )

        self.positions = torch.arange(start = 0, end = cf["num_patches"], step = 1, dtype = torch.int32)
        self.pos_embed = nn.Embedding(cf["num_patches"], cf["hidden_dim"])

       # transformer Encoder
        self.trans_encoder_layers = []

        for i in range(cf["num_layers"]):
            layer = nn.TransformerEncoderLayer(
                    d_model = cf["hidden_dim"],
                    nhead = cf["num_head"],
                    dim_feedforward = cf["mlp_dim"],
                    dropout = cf["dropout_rate"],
                    activation = nn.GELU(),
                    batch_first = True,
                    )
            self.trans_encoder_layers.append(layer)
        
        # CNN DeCoder 
        # Decoder 1
        self.d1 = DeConvBlock(cf["hidden_dim"], 512)
        self.s1 = nn.Sequential(
                DeConvBlock(cf["hidden_dim"], 512),
                ConvBlock(512, 512)
                )
        self.c1 = nn.Sequential( # Concatenation
                ConvBlock(512 + 512, 512),
                ConvBlock(512, 512)
                )

        # Decoder 2
        self.d2 = DeConvBlock(512, 256)
        self.s2 = nn.Sequential(
                DeConvBlock(cf["hidden_dim"], 256),
                ConvBlock(256, 256),
                DeConvBlock(256, 256),
                ConvBlock(256, 256),
                )
        self.c2 = nn.Sequential( # Concatenation
                ConvBlock(256 + 256, 256),
                ConvBlock(256, 256),
                )

        # Decoder 3
        self.d3 = DeConvBlock(256, 128)
        self.s3 = nn.Sequential(
                DeConvBlock(cf["hidden_dim"], 128),
                ConvBlock(128, 128),
                DeConvBlock(128, 128),
                ConvBlock(128, 128),
                DeConvBlock(128, 128),
                ConvBlock(128, 128),

                )
        self.c3 = nn.Sequential( # Concatenation
                ConvBlock(128 + 128, 128),
                ConvBlock(128, 128),
                )

        # Decoder 4
        self.d4 = DeConvBlock(128, 64)
        self.s4 = nn.Sequential(
                ConvBlock(3, 64),
                ConvBlock(64, 64),
                )
        self.c4 = nn.Sequential(
                ConvBlock(64 + 64, 64),
                ConvBlock(64, 64),
                )

        # output layer
        self.output = nn.Conv2d(64, 1, kernel_size = 1, padding = 0)


    def forward(self, inputs):
        # Path + Position Embeddings
        patch_embed = self.patch_embed(inputs) 

        positions = self.positions
        pos_embed = self.pos_embed(positions)
        
        x = patch_embed + pos_embed

        # transformer encoder
        skip_connection_index = [3, 6, 9, 12]
        skip_connections = []
        for i in range(self.cf["num_layers"]):
            layer = self.trans_encoder_layers[i]
            x = layer(x)

            if (i+1) in skip_connection_index:
                skip_connections.append(x)

        # CNN Decoder
        z3, z6, z9, z12 = skip_connections
        # print('[INFO] Input and middle layer shapes\n', inputs.shape, z3.shape, z6.shape, z9.shape, z12.shape)

        # Reshaping
        batch = inputs .shape[0]
        z0 = inputs.view((batch, self.cf["num_channels"], self.cf["image_size"], self.cf["image_size"]))
        # print("[INFO] Shape of z0:", z0.shape)

        shape = (batch, self.cf["hidden_dim"], self.cf["patch_size"], self.cf["patch_size"])
        z3 = z3.view(shape)
        z6 = z6.view(shape)
        z9 = z9.view(shape)
        z12 = z12.view(shape)
        # print("[INFO] z12 shape:", z12.shape) 

        # Decoder 1
        x = self.d1(z12)
        s = self.s1(z9)
        x = torch.cat([x, s], dim = 1)
        x = self.c1(x)
        # print("[INFO] Feature shape at Decoder 1:", x.shape)

        # Decoder 2
        x = self.d2(x)
        s = self.s2(z6)
        x = torch.cat([x, s], dim = 1)
        x = self.c2(x)
        # print("[INFO] Feature shape at Decoder 2:", x.shape)

        # Decoder 3
        x = self.d3(x)
        s = self.s3(z3)
        x = torch.cat([x, s], dim = 1)
        x = self.c3(x)
        # print("[INFO] Feature shape at Decoder 3:", x.shape)

        # Decoder 4
        x = self.d4(x)
        s = self.s4(z0)
        x = torch.cat([x, s], dim = 1)
        x = self.c4(x)
        # print("[INFO] Feature shape at Decoder 4:", x.shape)

        # output
        out = self.output(x)
        return out

if __name__ == "__main__":
    config = {}
    config["image_size"] = 256
    config["num_layers"] = 12
    config["hidden_dim"] = 768 
    config["mlp_dim"] = 3072
    config["num_head"] = 12 
    config["dropout_rate"]  =0.1 
    config["num_patches"] = 256 
    config["patch_size"] = 16
    config["num_channels"] = 3
    
    # original x
    x = torch.randn((
        8,
        config["num_patches"],
        config["patch_size"] * config["patch_size"] * config["num_channels"],
        ))

    print(f"[INFO] Shape of input: {x.shape}")

    model = UNETR_2D(config)
    model(x)
    y = model(x)
    print("[INFO] Output shape:", y.shape)

