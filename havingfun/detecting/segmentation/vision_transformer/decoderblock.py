import torch
import torch.nn as nn
from embeddingblock import EmbeddingBlock
from selfattentionblock import SelfAttentionBlock

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(DecoderBlock, self).__init__()

        self.embed_size = embed_size

        self.selfattention = SelfAttentionBlock(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tar_values, tar_keys, tar_queries, tar_mask):
        '''
        params
        ------
        input of self-attention block.
            values: torch.tensor, shape([N, values_len, embed_size])
            keys: torch.tensor, shape([N, key_len, embed_size])
            queries: torch.tensor, shape([N, queries_len, embed_size])
            mask: torch.tensor, shape([N, 1, 1, queries_len])

        returns
        ------
            out: torch.tensor, shape([N, queires_len, embed_size])
        '''
        tar_attention = self.selfattention(tar_values, tar_keys, tar_queries, tar_mask)
        out_decoderbock = self.dropout(self.norm(tar_attention + tar_queries))
        return out_decoderbock

# test this block
if __name__ == "__main__":
    source_vocab_size = 10
    source_padding_idx = 0

    target_vocab_size = 10
    target_padding_idx = 0

    max_length = 100
    dropout = 0.
    embed_size = 256
    heads = 8
    head_dim = embed_size // heads
    forward_wxpansion = 4
    
    num_layers = 6
    
    # input of decoderblock
    y = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]])
    tar = y
    print('======> target shape', tar.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_model = EmbeddingBlock(target_vocab_size, embed_size, max_length, dropout, device).to(device)
    decoder_model = DecoderBlock(embed_size, heads, dropout)
    def make_target_mask(tar_source):
        '''
        params
        ------
        tar_source: torch.tensor, shape([N, queries_len, embed_size])

        returns
        ------
        tar_mask: torch.tensor, shape([N, 1, target_len, target_len])
        '''
        N, tar_len = tar_source.shape
        tar_mask = torch.tril(torch.ones((tar_len, tar_len))).expand(N, 1, tar_len, tar_len)
        return tar_mask
    
    embedded_tar = embedding_model(tar)
    tar_values = embedded_tar
    tar_keys = embedded_tar
    tar_queries = embedded_tar
    mask_tar = make_target_mask(tar).to(device)
    print('======> mask_tar shape', mask_tar.shape)
    
    out_decoderblock = decoder_model(tar_values, tar_keys, tar_queries, mask_tar)
    print('======> decoderblock output shape', out_decoderblock.size())
