import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CONTEXT_SIZE = 128


class Embedding(nn.Module):
    """ Embeddings """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Multiply by sqrt(d_model) to scale the embedding
        out = self.embedding(x) * np.sqrt(self.d_model) # [batch, seq_len, d_model]
        return out
    

class PositionalEncoding(nn.Module):
    """ Positional encoding using sine and cosine functions """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        pe = torch.zeros(CONTEXT_SIZE, d_model)
        pos = torch.arange(0, CONTEXT_SIZE).unsqueeze(1)
        div_term = 1 / (10000 ** (torch.arange(0, d_model, 2) / d_model))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0) # [seq_len, d_model] -> [batch, seq_len, d_model]
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + self.pe[:, :x.shape[1], :].requires_grad(False) # [batch, seq_len, d_model]
        out = self.dropout(out) # [batch, seq_len, d_model]
        return out


class Head(nn.Module):
    """ One head of Self-Attention """
    def __init__(self, d_model, head_size, dropout=0.1):
        super().__init__()
        self.w_q = nn.Linear(d_model, head_size, bias=False)
        self.w_k = nn.Linear(d_model, head_size, bias=False)
        self.w_v = nn.Linear(d_model, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(CONTEXT_SIZE, CONTEXT_SIZE)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        q = self.w_q(query) # [batch, seq_len, head_size]
        k = self.w_k(key) # [batch, seq_len, head_size]
        v = self.w_v(value) # [batch, seq_len, head_size]

        # Compute attention scores
        a_s = (q @ k.transpose(-2, -1)) / np.sqrt(query.shape[-1]) # [batch, seq_len, seq_len]

        # Apply mask for Masked Self-Attention
        if mask == True:
            a_s = a_s.masked_fill_(self.tril[:a_s.shape[1], :a_s.shape[1]] == 0, float('-inf')) # [batch, seq_len, seq_len]
        
        a_s = F.softmax(a_s, dim=-1) # [batch, seq_len, seq_len]
        a_s = self.dropout(a_s) # [batch, seq_len, seq_len]
        out = a_s @ v # [batch, seq_len, head_size]
        return out


class MultiHeadAttention(nn.Module):
    """ Multi-Head Self-Attention """
    def __init__(self, d_model, n_heads, mask=False, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads.'

        head_size = d_model // n_heads
        self.heads = nn.ModuleList([Head(d_model, head_size, mask) for _ in range(n_heads)])
        self.linear = nn.Linear(n_heads * head_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        out = torch.cat([head(query, key, value, None) for head in self.heads], dim=-1) # [batch, seq_len, d_model]
        out = self.dropout(self.linear(out)) # [batch, seq_len, d_model]
        return out


class LayerNormalization(nn.Module):
    """ Layer Normalization """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # [batch, seq_len, 1]
        std = x.std(-1, keepdim=True) # [batch, seq_len, 1]
        out = self.weight * ((x + mean) / (std + self.eps)) + self.bias # [batch, seq_len, d_model]
        return out


class FeedForward(nn.Module):
    """ Position-wise Feed-Forward Network """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.relu(self.linear_1(x)) # [batch, seq_len, d_ff]
        out = self.linear_2(self.dropout(out)) # [batch, seq_len, d_model]
        return out


class ResidualConnection(nn.Module):
    """ Residual Connection """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        out = self.norm(x + self.dropout(sublayer(x))) # [batch, seq_len, d_model]
        return out
    

class Encoder(nn.Module):
    """ 
    Ecoder block: includes 2 sub-layers with residual connection and layer normalization:
        1. Multi-Head Attention
        2. Feed-Forward
    """
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.res1 = ResidualConnection(d_model)
        self.res2 = ResidualConnection(d_model)

    def forward(self, x):
        out = self.res1(out, self.multihead_attn(x, x, x)) # [batch, seq_len, d_model]
        out = self.res2(out, self.ff(out)) # [batch, seq_len, d_model]
        return out
    

class Decoder(nn.Module):
    """ 
    Decoder block: includes 3 sub-layers with residual connection and layer normalization:
        1. Masked Multi-Head Attention
        2. Multi-Head Attention
        3. Feed-Forward
    """
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(d_model, n_heads)
        self.masked_multihead_attn = MultiHeadAttention(d_model, n_heads, mask=True)
        self.ff = FeedForward(d_model, d_ff)
        
        self.res1 = ResidualConnection(d_model)
        self.res2 = ResidualConnection(d_model)
        self.res3 = ResidualConnection(d_model)

    def forward(self, x, enc_out=None):
        out = self.res1(out, self.masked_multihead_attn(x, x, x)) # [batch, seq_len, d_model]
        if enc_out is not None:
            out = self.res2(out, self.multihead_attn(enc_out, enc_out, out)) # [batch, seq_len, d_model]
        else:
            # for decoder-only models
            out = self.res2(out, self.multihead_attn(out, out, out)) # [batch, seq_len, d_model]
        out = self.res3(out, self.ff(out)) # [batch, seq_len, d_model]
        return out
    

class ProjectionLayer(nn.Module):
    """ Fully-connected layer followed by softmax activation function"""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        out = self.proj(x) # [batch, seq_len, vocab_size]
        # return the probabilities of predicted tokens
        out = F.softmax(out, dim=-1) # [batch, seq_len, vocab_size]
        return out