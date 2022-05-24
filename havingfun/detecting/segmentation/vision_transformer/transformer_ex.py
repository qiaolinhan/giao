# 2022-04-25

from turtle import forward, position
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

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size) # fully covolusional layer

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
            mask: torch.tensor

        returns
        ------
            outputs: torch.tensor
        ''' 
        N = queries.shape[0]
        values_len, keys_len, queries_len = values.shape[1], keys.shape[1], queries.shape[1]

        # split embedding into self.heads pieces
        values = values.reshape(N, values_len, self.heads, self.head_dim)
        keys = keys.reshape(N, keys_len, self.heads, self.head_dim)
        queries = queries.reshape(N, queries_len, self.heads, self.head_dim)

        values = self.values(values)
        print(f'======> 1 values shape: {values.shape}')
        keys = self.keys(keys)
        # print(f'=====> keys shape: {keys.shape}')
        queries = self.queries(queries)
        # print(f'======> queries shape: {queries.shape}')

        # 'n' for the N, 'h' for the heads, 'q' for the queries_len, 'd' for the head_dim 
        energy = torch.einsum("nqhd, nkhd -> nhqk", queries, keys)
        '''
        queries shape: (N, queries_len, heads, head_dim);
        keys shape: (N, keys_len, heads, head_dim);
        energy shape: (N, heads, queries_len, keys_len)
        '''
        print('======> energy shape', energy.shape)
        print('======> get in mask.shape', mask.shape)
        if mask is not None:
            energy = energy.masked_fill_(mask == 0, float("-1e20")) # mask == 0: shut that off, so that it does not impact any other.

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
        print("heads", heads)
        print("embed_size", embed_size)
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
        source_vocab_size, # original input size
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
        self.word_embeding = nn.Embedding(source_vocab_size, embed_size)
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
        # print('======> shape of positions', positions.shape)

        word_embedded_x = self.word_embeding(x)
        # print('======> shape of word embedded x', embedded_x.shape)
        ################
        out = self.dropout(self.word_embeding(x) + self.position_embeding(positions))
        # print ('======> shape of positional embedded x', out.shape)
        for layer in self.layers:
            out = layer(out, out, out, mask) # q, k, v, mask   
        print('======> shape of out', out.shape)        
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.device = device ####
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, source_mask, target_mask):
        print('======> shape of x', x.shape)
        print('======> shape of value', value.shape)
        print('======> shape of source_mask', source_mask.shape)
        print('======> shape of target mask', target_mask.shape)
        attention  = self.attention(x, x, x, target_mask)
        print('======> decoderblock attention size', attention.shape)
        query = self.dropout(self.norm(attention + x))
        print('======> decoderblock query size', query.shape)
        out = self.transformer_block(value, key, query, source_mask)
        print('======> shape out out of decoder block', out.shape)
        return out # the query to transformer block of decoder
 
class Decoder(nn.Module):
    def __init__(self, 
        mask_vocab_size, 
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
        self.word_embedding = nn.Embedding(mask_vocab_size, embed_size)
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
        source_vocab_size, 
        target_vocab_size, 
        source_pad_idx, 
        target_pad_idx,
        embed_size = 256,
        num_layer = 6,
        forward_expansion = 4,
        heads = 8,
        dropout = 0,
        device = "cuda",
        max_length = 100,
        ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(source_vocab_size, embed_size, num_layer, heads, forward_expansion, dropout, max_length, device)
        self.decoder = Decoder(target_vocab_size, embed_size, num_layer, heads, forward_expansion, dropout, max_length, device)
        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device
        print(f'======> model info:\n * num_layers: {num_layer}\n * embed_size: {embed_size}\n * forward_expansion: {forward_expansion}\n * max_length: {100}\n * dropout: {dropout}')
    
    # function to make source mask and target mask
    ################################
    def make_source_mask(self, source):
        source_mask = (source != self.source_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask shape: (N, 1, 1, src_len) 
        return source_mask.to(self.device)

    def make_target_mask(self, target):
        N, target_len = target.shape
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand(
            N, 1, target_len, target_len
        )
        return target_mask.to(self.device)
    ##################################
    def forward(self, source, target):
        source_mask = self.make_source_mask(source)
        target_mask = self.make_target_mask(target)
        encoder_source = self.encoder(source, source_mask)
        out = self.decoder(target, encoder_source, source_mask, target_mask)
        return out

# small example
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # embed_size = 256
    # heads = 8
    # head_dim = embed_size // heads
    # num_layers = 6
    # forward_expassion = 4
    # max_length = 100

    x = torch.tensor([[1, 5 ,6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    print('======> x.shape', x.shape)
    tar = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    print('======> target_mask input', tar[:, :-1])
    src_padding_idx = 0
    trg_padding_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_padding_idx, trg_padding_idx).to(device)
    
    out = model(x, tar[:, :-1])
    # out = model(x, mask)
    print(out.shape)