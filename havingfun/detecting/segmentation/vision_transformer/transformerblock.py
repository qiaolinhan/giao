import torch
import torch.nn as nn
from embeddingblock import EmbeddingBlock
from selfattentionblock import SelfAttentionBlock

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.selfattention_model = SelfAttentionBlock(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

 
    def forward(self, values, keys, queries, mask):
        '''
        params
        ------
        input of the self-attention block. Actually, query is especially involved in the computation of feedforward
            values: torch.tensor, shape([N, values_len, embed_size])
            keys: torch.tensor, shape([N, keys_len, embed_size])
            queries: torch.tensor, shape([N, queries_len, embed_size])
            mask: torch.tensor, shape([N, 1, 1, queries_len])

        returns
        ------
        out: torch.tensor, shape([N, queries_len, embed_size])
        '''
        attention = self.selfattention_model(values, keys, queries, mask)

        # add & norm: skip connection of [input of attention] and [output of attention]
        result_attention = self.dropout(self.norm1(attention + queries))
        input_feedforward = result_attention
        output_feedforward = self.feedforward(input_feedforward)
        out = self.dropout(self.norm2(input_feedforward + output_feedforward))
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
        return source_mask
    ################################    
    mask_source = make_source_mask(source).to(device)
    print('======> mask_source shape', mask_source.shape)

    transformerblock_model = TransformerBlock(embed_size, heads, dropout, forward_expansion).to(device)
    out = transformerblock_model(values_x, keys_x, queries_x, mask_source)
    print('======> transformerblock output shape', out.shape)
