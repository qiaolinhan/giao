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
from encoders import Encoders
from decoderblock import DecoderBlock
from transformerblock import TransformerBlock

class TransformerEntire(nn.Module):
    def __init__(self, 
            source_vocab_size, target_vocab_size,
            source_padding_idx, target_padding_idx,
            embed_size, heads,
            forward_expansion,
            dropout,
            device = device,
            max_length = 100,
            ):

        super(TransformerEntire, self).__init__()
        
        # encoder_embedding and decoder_embedding
        self.encoder_embedding = EmbeddingBlock(source_vocab_size,
                                                embed_size,
                                                max_length,
                                                dropout,
                                                device).to(self.device)
        self.decoder_embedding = EmbeddingBlock(target_vocab_size,
                                                embed_size,
                                                max_length,
                                                dropout,
                                                device).to(self.device)
        
        self.encoders = Encoders(embed_size, heads, num_layers, dropout, forward_expansion, source_padding_idx, device)
        self.decoderblock = DecoderBlock(embed_size, heads, dropout)
        self.transformerblock = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        
        # transformerlock of encoder part
        self.encoder_layers = nn.ModuleList(
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
                )
        # transformerblock of decoder part
        self.decoder_layers = nn.ModuleList(
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
                )
        # self.tar_padding_idx = tar_padding_idx

    def make_source_mask(self, source):
        source_mask = (source != self.source_padding_idx).unsqueeze(1).unsqueeze(2)
        return source_mask

    def make_target_mask(self, target):
        N, tar_len = target.shape
        tar_mask = torch.tril(torch.ones((tar_len, tar_len))).expand(N, 1,
                tar_len, tar_len)
        return tar_mask

    def forward(self, x, y):
        source = x
        target = y

        # inputs to encoder block
        embedded_source = self.encoder_embedding(source) 
        print('======> shape of embedded source to attention block of encoder',
                    embedded_source.shape)
        source_values = embedded_source
        source_keys = embedded_source
        source_queries = embedded_source
        source_mask = self.make_source_mask(source)
        print('======> shape of mask into encoder')
        # inputs to decoder block
        embedded_tar = self.decoder_embedding(target)
        print('======> shape of embedded target to attention blockof decoder',
                    embedded_tar.shape)
        tar_values = embedded_tar
        tar_keys = embedded_tar
        tar_queries = embedded_tar
        tar_mask = self.make_target_mask(target).to(self.device)
        print('======> shape of mask into decoderblock')
        
        # outouts of encoders and decodeblock
        # output of decoderblock
        queries_decoder = self.decoderblock(tar_values, tar_keys, tar_queries, tar_mask)
        # output of encoders
        output_encoder = self.encoders(source_values, source_keys, source_queries, source_mask)
        values_encoder = output_encoder
        keys_encoder = output_encoder

        for encoder_layer in self.encoder_layers:
            encoder_out = encoder_layer(source_values, source_keys, source_queries, source_mask)
            source_values = encoder_out
            source_keys = encoder_out
            source_queries = encoder_out
            # source_mask = self.make_source_mask(encoder_out)

        for decoder_layer in self.decoder_layers:
            decoder_out = decoder_layer(values_encoder, keys_encoder, queries_decoder, tar_mask)
            tar_values = decoder_out
            tar_keys = decoder_out
            tar_queries = decoder_out
            # tar_mask = self.make_target_mask(decoder_out)

        out = decoder_out
        return out


if __name__ == '__main__':
    source_vocab_size = 10
    source_padding_idx = 0

    target_vocab_size = 10
    target_padding_idx = 0
    
    max_length = 100
    dropout = 0.
    embed_size = 256
    heads = 8

    head_dim = embed_size // heads
    forward_expansion = 4

    num_layers = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = torch.tensor([[1, 5 ,6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    y = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2 ]]).to(device)
    print('======> input source shape', x.shape)
    print('======> input target shape', y.shape)
    transformer_model = TransformerEntire(source_vocab_size, target_vocab_size, max_length, embed_size,
            heads, num_layers, dropout, forward_expansion, source_padding_idx, device) 
    output_transformer = transformer_model(x, y)
    print('======> output shape', output_transformer.shape)
