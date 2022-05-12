import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(DecoderBlock, self).__init__()
        