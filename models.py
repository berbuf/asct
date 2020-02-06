# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from revnet import ReversibleBlock, ReversibleSequence

#from adaptive_span import AdaptiveSpan

# Size notations:
# B = batch_size, H = hidden_size, M = block_size, L = attn_span

class SelectAttention(nn.Module):

    def __init__(self, batch_size, n_heads, max_span):
        nn.Module.__init__(self)
        B, K, M = batch_size, n_heads, max_span

        self.n_heads = K
        mask = (torch.linspace(0, M - 1, M)
                .repeat(B * K * M, 1)
                .reshape(B, K, M, M)
                - torch.arange(M).float().unsqueeze(1)).cuda()
        self.register_buffer('mask', mask)
        self.soft = 2

    def forward(self, attn, span):
        (B,M,_), K = attn.size(), self.n_heads
        
        attn = attn.reshape(B//K, K, M, -1)
        span = span.reshape(B//K, K, M, 2, 1)
        
        mean = span[:,:,:,1]
        intercept = span[:,:,:,0]
        """
        print (attn.shape)
        print (span.shape)
        print (mean.shape)
        print (intercept.shape)
        print (self.mask.shape)
        """

        # -((x+b)/soft)**2+a
        z = -((self.mask - intercept) / self.soft)**2 + mean
        z = z.clamp(0, 1)

        attn = attn * z

        return attn.reshape(B,M,M)

class SelfAttention(nn.Module):
    """ """
    def __init__(self, dup_batch_size, hidden_size, block_size,
                 dropout, nb_heads, **kargs):
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size # size of a single head
        # B K M
        self.select = SelectAttention(dup_batch_size, nb_heads, block_size)

    def forward(self, query, key, span, value, key_pe):
        B,M,H=key.size()

        attn_cont = torch.matmul(query, key.transpose(-1, -2))
        attn_pos = torch.matmul(query, key_pe)
        attn = attn_cont + attn_pos

        # select
        attn = self.select(attn, span)

        attn = attn / math.sqrt(H)
        attn = F.softmax(attn, dim=-1)

        attn = self.dropout(attn)
        
        out = torch.matmul(attn_cont, value)
        
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, nb_heads, **kargs):
        nn.Module.__init__(self)
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn = SelfAttention(
            hidden_size=self.head_dim, nb_heads=nb_heads, **kargs)
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

    def forward(self, h, key_pe):
        B = h.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = h.size(1)

        query = self.head_reshape(self.proj_query(h), self.head_dim)
        key = self.head_reshape(self.proj_key(h), self.head_dim)
        span = self.head_reshape(self.proj_span(h), 2)
        value = self.head_reshape(self.proj_val(h), self.head_dim)

        out = self.attn(query, key, span, value, key_pe)  # B_K x M x D
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        return out

# Boom layer
class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, inner_hidden_size, dropout, **kargs):
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
    def __init__(self, hidden_size, **kargs):
        nn.Module.__init__(self)
        self.attn = MultiHeadSelfAttention(hidden_size=hidden_size, **kargs)
        self.ff = FeedForwardLayer(hidden_size=hidden_size, **kargs)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, h, key_pe):
        # h = B x M x H
        attn_out = self.attn(h, key_pe)
        h = self.norm1(h + attn_out)  # B x M x H
        ff_out = self.ff(h)
        out = self.norm2(h + ff_out)  # B x M x H
        return out

class SideRevNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fn = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.fn(x)
        x = F.gelu(x)
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, nb_heads, nb_layers,
                 block_size, rev_net, **kargs):
        nn.Module.__init__(self)
        # token embeddings
        self.in_emb = nn.Embedding(vocab_size, hidden_size)
        self.out_emb = nn.Linear(hidden_size, vocab_size)
        # position embeddings
        self.key_pe = nn.Parameter(
            torch.randn(1, hidden_size // nb_heads, block_size))
        # reversible blocks
        rev_block = (lambda x: x if not rev_net else
                     ReversibleBlock(f_block=x,
                                     g_block=SideRevNet(hidden_size=hidden_size),
                                     split_along_dim=-1))
        # transformer layers
        self.layers = nn.ModuleList([
            rev_block(TransformerLayer(hidden_size=hidden_size,
                                       nb_heads=nb_heads,
                                       block_size=block_size,
                                       **kargs))
            for _ in range(nb_layers) ])
        if rev_net:
            self.layers = ReversibleSequence(self.layers)

    def forward(self, x):
        # project into embeddings
        h = self.in_emb(x)  #  B x M => B x M x H
        # iter on layers
        for l, layer in enumerate(self.layers):
            h = layer(h, self.key_pe)  # B x M x H
        # decoder
        out = F.log_softmax(self.out_emb(h), dim=-1)
        return out
