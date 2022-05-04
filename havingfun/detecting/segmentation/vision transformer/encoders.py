from sqlalchemy import values
import torch
import torch.nn as nn
from embeddingblock import EmbeddingBlock
from transformerblock import TransformerBlock

'''
params
------
v, k, q, mask
v: values, torch.tensor, shape([N, values_len, embed_size])
k: keys, torch.tensor, shape([N, keys_len, embed_size])
q: queries, torch.tensor, shape([N, queries_len, embed_size])
mask: mask_source, torch.tensor, shape([N, 1, 1, queries_len])

returns
------
output: input to decoder(s), shape([N, queries_len, embed_size])
'''

class Encoders(nn.Module):
    def __init__(self, embed_size, heads, num_layers, dropout, forward_expansion, source_padding_idx, device): # params for embedding: (source_vocab_size, max_length, device) 
        super(Encoders, self).__init__()

        self.device = device
        self.embedding = EmbeddingBlock(source_vocab_size, embed_size, max_length, dropout, device).to(self.device)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)
            ]
            
        )
        self.source_padding_idx = source_padding_idx
    def make_source_mask(self, source):
        source_mask = (source != self.source_padding_idx).unsqueeze(1).unsqueeze(2)
        # src_mask shape: (N, 1, 1, src_len) 
        return source_mask.to(self.device)

    def forward(self, x, mask_source):
        source_x = self.embedding(x)
        values_x = source_x
        keys_x = source_x
        queries_x = source_x

        for layer in self.layers:
            out = layer(values_x, keys_x, queries_x, mask_source)
            # print('======> encoder output shape', out.size())
            values_x = out
            keys_x = out
            queries_x = out
            print('======> mask_source in compute shape', mask_source.size())
        
        return out

if __name__ == "__main__":
    # params for the model
    ################################
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

    num_layers = 6 # Nx, how many times the encoder/decoder will repeat

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ###################################
    # models to be tested
    embedding_model = EmbeddingBlock(source_vocab_size, embed_size, max_length, dropout, device).to(device)
    encoder_model = Encoders(embed_size, heads, num_layers, dropout, forward_expansion, source_padding_idx, device).to(device)

    x = torch.tensor([[1, 5 ,6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    source = x
    print('======> source shape', x.shape)
    # embedded_source = embedding_model(source)
    # values_x = embedded_source
    # keys_x = embedded_source
    # queries_x = embedded_source

    # function to make source mask
    ################################
    def make_source_mask(source):
        '''
        params
        ------
        source: torch.tensor, shape([N, queries_len, embed_size])

        returns
        ------
        source_mask: torch.tensor, shape([N, 1, 1, queries_len])
        '''
        source_mask = (source != source_padding_idx).unsqueeze(1).unsqueeze(2)
        # src_mask shape: (N, 1, 1, src_len) 
        return source_mask
    ################################
    mask_source = make_source_mask(source).to(device)
    print('======> mask_source shape', mask_source.shape)
    
    out = encoder_model(x, mask_source)
    print('======> encoder output shape', out.size())