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

class SeqAttention(nn.Module):
    """Sequential self-attention layer.
    Each token will attend to its previous fixed number of steps.
    Note that attention doesn't include the current step itself.
    """
    def __init__(self, hidden_size,
                 dropout, **kargs):
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size # size of a single head

    def forward(self, query, key, value, key_pe):
        # compute attention from context
        # B x M (dest) x (l+M) (src)
        attn_cont = torch.matmul(query, key.transpose(-1, -2))

        # compute the effect of position embedding
        attn_pos = torch.matmul(query, key_pe)  # B x M x L_pos

        B,M,_ = attn_pos.size()
        attn = attn_cont + attn_pos#[:,:,:M]

        attn = attn / math.sqrt(self.hidden_size)  # B x M X L_pos
        attn = F.softmax(attn, dim=-1)

        attn = self.dropout(attn)  # B x M X L_pos

        out = torch.matmul(attn_cont, value)  # B x M x H
        return out

class MultiHeadSeqAttention(nn.Module):
    def __init__(self, hidden_size, nb_heads, **kargs):
        nn.Module.__init__(self)
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn = SeqAttention(
            hidden_size=self.head_dim, nb_heads=nb_heads, **kargs)
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias=False)

    def head_reshape(self, x):
        # x: B x (L+M) x H
        K = self.nb_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (L+M) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (L+M) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (L+M) x D
        return x

    def forward(self, h, key_pe):
        B = h.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = h.size(1)

        query = self.head_reshape(self.proj_query(h))
        value = self.head_reshape(self.proj_val(h))
        key = self.head_reshape(self.proj_key(h))

        out = self.attn(query, key, value, key_pe)  # B_K x M x D
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


class TransformerSeqLayer(nn.Module):
    def __init__(self, hidden_size, **kargs):
        nn.Module.__init__(self)
        self.attn = MultiHeadSeqAttention(hidden_size=hidden_size, **kargs)
        self.ff = FeedForwardLayer(hidden_size=hidden_size, **kargs)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, h, key_pe):
        attn_out = self.attn(h, key_pe)
        h = self.norm1(h + attn_out)  # B x M x H
        ff_out = self.ff(h)
        out = self.norm2(h + ff_out)  # B x M x H
        return out


class Vanilla(nn.Module):
    def __init__(self, vocab_size, hidden_size,
                 nb_heads, nb_layers, **kargs):
        attn_span = 512
        nn.Module.__init__(self)
        self.key_pe = nn.Parameter(
            torch.randn(1, hidden_size // nb_heads, attn_span))
        self.layers = nn.ModuleList([
            TransformerSeqLayer(hidden_size=hidden_size,
                                nb_heads=nb_heads,                
                                **kargs)
            for _ in range(nb_layers) ])

    def forward(self, h, _):
        # iter on layers
        for l, layer in enumerate(self.layers):
            # forward
            h = layer(h, self.key_pe)  # B x M x H
        return h
