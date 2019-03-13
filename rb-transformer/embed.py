#!/usr/bin/python                                                                                                                                                   
import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class Embedder(nn.Module):
    def __init__(self, vocab_size, dim_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, dim_embed, max_seq_len = 80, dropout = 0.1):
        super().__init__()
        self.dim_embed = dim_embed
        self.dropout = nn.Dropout(dropout)

        # create constant 'pe' matrix with values dependant on pos and i                                                                                            
        pe = torch.zeros(max_seq_len, dim_embed)
        for pos in range(max_seq_len):
            for i in range(0, dim_embed, 2):
                val1 = 10000 ** ((2 * i)/dim_embed)
                val2 = 10000 ** ((2 * i + 1)/dim_embed)
                pe[pos, i] = math.sin(pos / val1)
                pe[pos, i + 1] = math.cos(pos / val2)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        # make embeddings relatively larger                                                                                                                         
        x = x * math.sqrt(self.dim_embed)
        # add constant to embedding                                                                                                                                  
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        x = x + pe
        return x
