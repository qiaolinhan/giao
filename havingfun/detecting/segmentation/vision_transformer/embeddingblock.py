import torch
import torch.nn as nn

'''
torch.nn.Embedding:
    params
    ------
    num_embeddings: int, num of words in dictionary;
    embedding_dim: int, dim of embedding;
    padding_idx: if exist, fill 0 at the position

    forward
    -----
    input: Longtensor, (*)

    returns
    -------
    output: torch.tensor, size:(*, embedding_dim)
'''

class EmbeddingBlock(nn.Module):
    def __init__(self, source_vocab_size, embed_size, max_length, dropout, device):
        super(EmbeddingBlock, self).__init__()

        self.embed_size = embed_size
        self.max_length = max_length
        self.word_embedding = nn.Embedding(source_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)
        self.dropout = nn.Dropout(dropout)

        # it is needed because of the positional embedding also need to be load into device ('cuda') 
        self.device = device
    def forward(self, x):
        word_embedding_x = self.word_embedding(x)
        N, sequence_len = x.shape
        positions = torch.arange(0, sequence_len).expand(N, sequence_len).to(self.device)
        positional_embedding_x = self.positional_embedding(positions)
        embedding_x = self.dropout(word_embedding_x + positional_embedding_x)
        return embedding_x

if __name__ == '__main__':
    source_vocab_size = 10
    embed_size = 256
    max_length = 100 
    dropout = 0.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([[1, 5 ,6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    print('======> x.size', x.size())

    # word_x_model = nn.Embedding(num_embeddings = source_vocab_size, embedding_dim = embed_size).to(device) # nn.Embedding(10, 256)
    # word_x = word_x_model(x)
    # print('======> word_x.size', word_x.size())

    # N, sequence_len = x.shape
    # positions = torch.arange(0, sequence_len).expand(N, sequence_len)
    # position_x_model = nn.Embedding(max_length, embed_size)
    # position_x = position_x_model(positions)
    # print('======> position_x.size', position_x.size())

    embedding_model = EmbeddingBlock(source_vocab_size, embed_size, max_length, dropout, device).to(device)
    embedding_x = embedding_model(x)
    print('======> emebdding_x.shape', embedding_x.shape)
