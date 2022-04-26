# 2022-04-25

from turtle import forward
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        '''
        Split embeddings into different parts, eg: embedding size(256), heads(8) -> 8 by 32 parts 
        
        Parameters
        ------
            embed_size: int, the size of the embedding; (256)
            heads: int, the number of parts would like to separate; (8)
            '''
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads == embed_size) # embed_size should be devide-able by heads

        # V, Q, K
        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size) # fully connected linear

    def forward(self, values, queries, keys, mask):
        '''
        split embedding into different parts
        Parameters
        ------
            values: torch.tensor,
                shape: (queries_shape[0], values_len, self.heads, self.head_dim)
            queries: torch.tensor,
                shape: (queries_shape[0], queries_len, self.heads, self.head_dim)
            keys: torch.tensor,
                shape: (queries_shape[0], keys_len, self.heads, self.head_dim)
            mask: torch.tensor,
                shape: ()

        returns
        ------
            outputs: torch.tensor,
                shape: ()
        '''
        N = queries.shape[0]
        values_len, keys_len, queries_len = values.shape[1], keys.shape[1], queries.shape[1]

        # split embedding into self.heads pieces
        values = values.reshape(N, values_len, self.heads, self.head_dim)
        keys = keys.reshape(N, keys_len, self.heads, self.head_dim)
        queries = queries.reshape(N, queries_len, self.heads, self.head_dim)

        values = self.values(values)
        # print(f'values shape: {values.shape()}')
        keys = self.keys(keys)
        # print(f'keys shape: {keys.shape()}')
        queries = self.queries(queries)
        # print(f'queries shape: {queries.shape()}')

        # 'n' for the N, 'h' for the heads, 'q' for the queries_len, 'd' for the head_dim 
        energy = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])
        '''
        queries shape: (N, queries_len, heads, head_dim);
        keys shape: (N, keys_len, heads, head_dim);
        energy shape: (N, heads, queries_len, keys_len)
        '''

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20")) # mask == 0: shut that off, so that it does not impact any other.

        # attention(V, K, Q) = softmax(Q  K^T / embed_size ** (1/ 2)) * V

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3)

        
        out = torch.einsum("nhql, nlhd -> nqhd", [attention, values]).reshape(
            N, queries_len, self.heads * self.head_dim
        )
        '''
        attention shape: (N, heads, queries_len, keys_len)
        values shape: (N, values_len, heads, head_dim)
        after einsum: (N, queires_len, heads, head_dim)
        '''

        out = self.fc_out(out)  
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # print('forward expansion', forward_expansion)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        ) # embed_size --> forward_expansion * embed_size --> embed_size

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feedforward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    # original input --> word embedding --> positional embedding --> [transformer block] --> values, keys
    def __init__(self,
        src_vocab_size, # original input size
        embed_size,  # size of embedded input
        num_layers,  # Nx
        heads, # embed_size // heads = head_dim
        forward_expansion,
        dropout,    # dropout(norm(a + b))
        max_length, # sequence_length
        device,
    ):
        super(Encoder, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.word_embeding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embeding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout = dropout,
                    forward_expansion=forward_expansion,
                )
            for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, sequence_len = x.shape
        ################ for postion embedding
        positions = torch.arange(0, sequence_len).expand(N, sequence_len).to(self.device)
        ################
        out = self.dropout(self.word_embeding(x) + self.position_embeding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask) # q, k, v, mask

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.device = device ####
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, value, key, src_mask, trg_mask):
        attention  = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, 
        trg_vocab_size, 
        embed_size, 
        num_layers, 
        heads, 
        forward_expansion, 
        dropout, 
        max_length, 
        device
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        position = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(position)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask) # (v, k, q)

        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(self, 
        src_vocab_size, 
        trg_vocab_size, 
        src_pad_idx, 
        trg_pad_idx,
        embed_size = 256,
        num_layer = 6,
        forward_expansion = 4,
        heads = 8,
        dropout = 0,
        device = "cuda",
        max_length = 100,
        ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layer, heads, forward_expansion, dropout, max_length, device)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layer, heads, forward_expansion, dropout, max_length, device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    # function to make source mask and target mask
    ################################
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask shape: (N, 1, 1, src_len) 
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)
    ##################################
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

# small example
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try_embed_size = 256
    try_heads = 8
    try_head_dim = try_embed_size // try_heads
    print(try_head_dim)

    x = torch.tensor([[1, 5 ,6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
    out = model(x, trg[:, :-1])
    print(out.shape)