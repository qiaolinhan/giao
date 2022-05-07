import torch
import torch.nn as nn
from embeddingblock import EmbeddingBlock
from selfattentionblock import SelfAttentionBlock

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(DecoderBlock).__init__()

        self.selfattention = SelfAttentionBlock(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        