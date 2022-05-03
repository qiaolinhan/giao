import torch
import torch.nn as nn
from embeddingblock import EmbeddingBlock
from transformerblock import TransformerBlock

class Encoders(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion): # params for embedding: (source_vocab_size, max_length, device) 
        super(Encoders, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
            ]
        )

    def forward(self, values, keys, queries, mask):
        pass

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ###################################
    embedding_model = EmbeddingBlock(source_vocab_size, embed_size, max_length, dropout, device).to(device)

    x = torch.tensor([[1, 5 ,6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    source = x
    print('======> source shape', x.shape)
    embedded_source = embedding_model(source)
    values_x = embedded_source
    keys_x = embedded_source
    queries_x = embedded_source

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
        return source_mask.to(device)
    ################################
    mask_source = make_source_mask(source)
    print('======> mask_source shape', mask_source.shape)
    encoder_model = Encoders(values_x, keys_x, queries_x, mask_source)