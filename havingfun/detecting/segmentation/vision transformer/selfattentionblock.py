import torch
import torch.nn as nn

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_size, heads, device):
        super(SelfAttentionBlock, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size // heads
        assert(self.heads_dim * heads == embed_size) # embed_size should be devide-able by heads

        self.get_values = nn.Linear(self.heads_dim, self.heads_dim, bias = False)
        self.get_keys = nn.Linear(self.heads_dim, self.heads_dim, bias = False)
        self.get_queries = nn.Linear(self.heads_dim, self.heads_dim, bias = False)

        self.device = device

        def get_source_mask(self, source):
            source_mask = (source != self.source_pad_idx).unsqueeze(1).unsqueeze(2)
            # src_mask shape: (N, 1, 1, src_len) 
            return source_mask.to(self.device)


    def forward(self, x):
        values = self.get_values(x)
        keys = self.get_keys(x)
        queries = self.get_queries(x)

        source_mask = self.get_source_mask(x)

        N = queries.shape[0]
        values_len, keys_len, queries_len = values.shape[1], keys.shape[1], queries.shape[1]




