#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: decoders.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-05-24
#
#   @Email: q_linhan@live.concordia.ca
#
#   @Description: This is the block to assign output of encoders and output of decoders
#                 for building transformer
#
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from embeddingblock import EmbeddingBlock
from decoderblock import DecoderBlock
from transformerblock import TransformerBlock

class Decoders(nn.Module):
    def __init__(self, embed_size, heads, num_layers, dropout, forward_expansion, tar_padding_idx, device):
        super(Decoders, self).__init__()
        
        self.dropout = dropout
        self.device = device
        self.decoderblock = DecoderBlock
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device),
            TransformerBlock(embed_size, heads, dropout, forward_expansion),
            for _ in range(num_layers)
            ]
        )
        self.tar_padding_idx = tar_padding_idx

    def make_target_mask(self, target):
        N, tar_len = target.shape
        tar_mask = torch.tril(torch.ones((tar_len, tar_len))).expand(N, 1,
                tar_len, tar_len)
        return tar_mask

    def forward(self, x):
        pass



