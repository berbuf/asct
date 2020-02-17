#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#from revnet import ReversibleBlock, ReversibleSequence
from asa import AutoSelectAttention
from act import AdaptiveComputationTime

# Size notations:
# B = batch_size, H = hidden_size, M = block_size, K = nb_heads

class SelfAttention(nn.Module):
    """ self-attention with selective attention """

    def __init__(self, hidden_size, dropout,
                 nb_heads, block_size, soft, **kargs):
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size # size of a single head
        # B K M
        self.select = AutoSelectAttention(nb_heads, block_size, soft)

    def forward(self, query, key, span, value):
        B,M,H=key.size()
        # compute attention value
        attn_cont = torch.matmul(query, key.transpose(-1, -2))
        # select attention
        attn = attn_cont
        attn = self.select(attn, span)
        # normalize
        attn = attn / math.sqrt(H)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        # project to inner dim
        out = torch.matmul(attn_cont, value)        
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, nb_heads, **kargs):
        nn.Module.__init__(self)
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn = SelfAttention(
            hidden_size=self.head_dim,
            nb_heads=nb_heads, **kargs)
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_span = nn.Linear(hidden_size, 2 * nb_heads, bias=False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias=False)

    def head_reshape(self, x, head_dim):
        # x: B x M x H
        K = self.nb_heads
        D = head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x M x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x M x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x M x D
        return x

    def forward(self, h):
        B = h.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = h.size(1)

        query = self.head_reshape(self.proj_query(h),
                                  self.head_dim)
        key = self.head_reshape(self.proj_key(h),
                                self.head_dim)
        span = self.head_reshape(self.proj_span(h), 2)
        value = self.head_reshape(self.proj_val(h),
                                  self.head_dim)

        out = self.attn(query, key, span, value)  # B_K x M x D
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        return out

# Boom layer
class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, inner_hidden_size,
                 dropout, **kargs):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(hidden_size, inner_hidden_size)
        self.fc2 = nn.Linear(inner_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h1 = F.relu(self.fc1(h))
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2

class TransformerLayer(nn.Module):
    def __init__(self, batch_size, block_size, hidden_size,
                 nb_heads, act, **kargs):
        nn.Module.__init__(self)
        self.attn = MultiHeadSelfAttention(
            block_size=block_size,
            hidden_size=hidden_size,
            nb_heads=nb_heads,
            **kargs)
        if act:
            self.act = AdaptiveComputationTime(
                batch_size, block_size,
                hidden_size, **kargs)
        self.ff = FeedForwardLayer(hidden_size=hidden_size,
                                   **kargs)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.act = act

    def forward(self, h):
        # h = B x M x H
        attn_out = self.attn(h)
        h = self.norm1(h + attn_out)  # B x M x H
        ff_out = self.ff(h)
        out = self.norm2(h + ff_out)  # B x M x H
        if self.act:
            out = self.act(out)
        return out

class Generator(nn.Module):
    def __init__(self, vocab_size, batch_size, hidden_size,
                 nb_heads, nb_layers, block_size, **kargs):
        nn.Module.__init__(self)
        # decoder
        self.out_emb = nn.Linear(hidden_size, vocab_size)
        # transformer layers
        self.layer = TransformerLayer(batch_size=batch_size,
                                      block_size=block_size,
                                      hidden_size=hidden_size,
                                      nb_heads=nb_heads,
                                      act=False,
                                      **kargs)
        self.nb_layers = nb_layers

    def forward(self, h):
        for _ in range(self.nb_layers):
            h = self.layer(h)  # B x M x H
        # decoder
        out = self.out_emb(h)
        return out

class Discriminator(nn.Module):
    def __init__(self, vocab_size, batch_size, hidden_size,
                 nb_heads, nb_layers, block_size, **kargs):
        nn.Module.__init__(self)
        # decoder
        self.out_emb = nn.Linear(hidden_size, block_size)
        # transformer layers
        self.layer = TransformerLayer(batch_size=batch_size,
                                      block_size=block_size,
                                      hidden_size=hidden_size,
                                      nb_heads=nb_heads,
                                      act=True,
                                      **kargs)

    def forward(self, h):
        # init act
        self.layer.act.init_act()
        # loop until empty
        _,M,_=h.size()
        while M:
            h = self.layer(h)  # B x M x H
            _,M,_=h.size()
        h = self.layer.act.weighted_h
        # decoder
        out = F.sigmoid(self.out_emb(h), dim=-1)
        return out

class GenDisc(nn.Module):
    def __init__(self, vocab_size, batch_size, model_params):
        nn.Module.__init__(self)
        # Shared token embeddings
        self.in_emb = nn.Embedding(vocab_size,
                                   model_params["hidden_size"])
        self.gen = Generator(vocab_size, batch_size,
                             **model_params)
        self.disc = Discriminator(vocab_size, batch_size,
                                  **model_params)

    def forward(self, x):
        h = self.in_emb(x)
        out_gen = self.gen(h)
        return None
        x_gen = decode(out_gen)
        h = self.in_emb(x_gen)
        out_disc = self.disc(h)
        return out_gen, x_gen, out_disc
