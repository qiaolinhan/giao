'''
einsum Q * K:
------
    params:
    ------
    queries shape: (N, queries_len, heads, head_dim);
    keys shape: (N, keys_len, heads, head_dim);

    returns:
    ------
    energy shape: (N, heads, queries_len, keys_len)

einsum energy * V:
------
    params:
    ------
    energy shape: (N, heads, queries_len, keys_len);
    values shape: (N, values_len, heads, head_dim);

    returns:
    ------
    after einsum: (N, queires_len, heads, head_dim)
'''

import torch
import torch.nn as nn

# in example
from embeddingblock import EmbeddingBlock

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_size, heads, device):
        super(SelfAttentionBlock, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size // heads
        assert (self.heads_dim * heads == embed_size) , "embed_size should be divisible by heads."

        self.get_values = nn.Linear(self.heads_dim, self.heads_dim, bias = False)
        self.get_keys = nn.Linear(self.heads_dim, self.heads_dim, bias = False)
        self.get_queries = nn.Linear(self.heads_dim, self.heads_dim, bias = False)
        self.fc_out = nn.Linear(heads * self.heads_dim, embed_size)

        self.device = device

    def forward(self, values_x, keys_x, queries_x, mask):
        # split embedding into self.heads pieces
        N = queries_x.shape[0]
        values_len, keys_len, queries_len = values_x.shape[1], keys_x.shape[1], queries_x.shape[1]
        values_x = values_x.reshape(N, values_len, self.heads, self.heads_dim)
        keys_x = keys_x.reshape(N, keys_len, self.heads, self.heads_dim)
        queries_x = queries_x.reshape(N, queries_len, self.heads, self.heads_dim)

        # getting v, k, q from x
        values = self.get_values(values_x)
        keys = self.get_keys(keys_x)
        queries = self.get_queries(queries_x)

        '''
        Using torch.einsum to compute tensors multiplication
            params
            ------
            queries (N, queries_len, heads, heads_dim) --> (nqhd)
            keys (N, keys_len, heads, heads_dim) --> (nkhd)
            
            returns
            ------
            energy (N, heads, queries_len, keys_len) --> (nhqk)
        '''
        energy = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill_(mask == 0, float("-1e20")) # mask == None: shut that off, so that it does not impact any other.
            
        # attention(V, K, Q) = softmax(Q  K^T / embed_size ** (1/ 2)) * V

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3)
        attention_out = torch.einsum("nhqk, nvhd -> nqhd", [attention, values]).reshape(N, queries_len, self.heads * self.heads_dim)
        out = self.fc_out(attention_out)  
        return out

if __name__ == "__main__":
    source_vocab_size = 10
    max_length = 100 
    dropout = 0.
    embed_size = 256
    heads = 8
    head_dim = embed_size // heads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([[1, 5 ,6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    print('======> x.shape', x.shape)


    embedding_model = EmbeddingBlock(source_vocab_size, embed_size, max_length, dropout, device).to(device)
    values_x = embedding_model(x)
    keys_x = embedding_model(x)
    queries_x = embedding_model(x)

    target = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device) # bottom input of decoder
    print('======> target_mask.shape', target.shape)
    print('======> target_mask input', target[:, :-1])
    embedded_target = embedding_model(target[:, :-1])
    print('======> target_f shape', embedded_target.size())

    # function to make source mask and target mask
    ################################
    def make_source_mask(self, source):
        source_mask = (source != self.source_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask shape: (N, 1, 1, src_len) 
        return source_mask.to(self.device)

    def make_target_mask(self, embedded_target):
        N, target_len = embedded_target.shape
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand(
            N, 1, target_len, target_len
        )
        return target_mask.to(self.device)
    ##################################

    target_mask = make_target_mask(embedded_target)
    print('======> targe_mask shape', target_mask.shape)

    selfattention_model = SelfAttentionBlock(embed_size, heads, device).to(device)
    selfattention_out = selfattention_model(values_x, keys_x, queries_x, mask = target_mask)
    print('======> selfattention output shape', selfattention_out.size())



