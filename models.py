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

from adaptive_span import AdaptiveSpan

# Size notations:
# B = batch_size, H = hidden_size, M = block_size, L = attn_span

def _skew(X, pad_value):
    """shift every row 1 step to right"""
    """ realign values, pad non-relevant span values with pad_value"""
    # X = B x M x L
    B, M, L = X.size()
    X = F.pad(X, (0, M + 1), value=pad_value)  # B x M x (L+M+1)
    X = X.view(B, -1)  # B x LM+MM+M
    X = X[:, :-M]  # B x LM+MM
    X = X.view(B, M, L + M)  # B x M x L+M
    return X

def _unskew(X):
    """reverse _skew operation"""
    """ crop non-relevant span """
    # X = B x M x L+M
    B, M, L_M = X.size()
    L = L_M - M
    X = X.view(B, -1)  # B x LM+MM
    X = F.pad(X, (0, M))  # B x LM+MM+M
    X = X.view(B, M, L + M + 1)  # B x M x L+M+1
    X = X[:, :, :L]  # B x M x L
    return X

def _span_slice(X, span):
    """ get slices of X according to span """
    B, M, H = X.size()

    # window informations
    left_bound = (torch.max(span[:,:,0], -span[:,:,1])
                  .unsqueeze(2).round().int())
    right_bound = (torch.max(span[:,:,0], -span[:,:,1])
                   .unsqueeze(2).round().int())
    max_span = ((left_bound + right_bound)
                .abs().max().long().item()) + 1 # central token

    # basic span index of max_span size
    idx_span = ((torch.arange(max_span)
                 # one idx for each token
                 .unsqueeze(1).expand(-1, B * M)
                 # format to rows
                 .transpose(0, 1).view(B, M, max_span)
                 # align index with current token
                 + torch.arange(M).unsqueeze(1))
                # align index with actual span start for each token
                .cuda() - left_bound)

    # index out of window indexes to idx_pad 
    idx_pad = M
    idx_span[idx_span < 0] = idx_pad
    right_mask = (right_bound + torch.arange(M).unsqueeze(1).cuda())
    idx_span[idx_span > right_mask] = idx_pad

    # extract indexes and project values to third dim
    key_span = F.pad(X, (0, 0, 0, 1))
    return (key_span
            .view(-1, H)[idx_span.view(-1, max_span)]
            .view(B, M, max_span, H)
            .contiguous())

class SelfAttention(nn.Module):
    """ """
    def __init__(self, hidden_size, attn_span,
                 dropout, adapt_span_params, **kargs):
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size # size of a single head
        self.attn_span = attn_span
        self.adapt_span_enabled = adapt_span_params['adapt_span_enabled']
        self.zeros = torch.zeros(64, attn_span, attn_span, hidden_size)
        if self.adapt_span_enabled:
            self.adaptive_span = AdaptiveSpan(attn_span=attn_span,
                                              **adapt_span_params, **kargs)

    def forward(self, query, key, span, value, key_pe):
        # query, key, value size = B x M x H
        # key_pe = B x M x L
        B, M, H = query.size()

        if self.adapt_span_enabled:
            # [optional] trim out memory to reduce unnecessary computation
            key, value, key_pe = self.adaptive_span.trim_memory(
                query, key, value, key_pe)

        # Query * Key^T
        K = _span_slice(key, span)
        print (query.view(B, M, 1, H).shape)
        print (K.shape)
        attn_cont = (torch.matmul(
            query.view(B, M, 1, H), K.transpose(-1, -2))
                     .reshape(B, M, -1)) # B, M, S
        print (attn_cont.shape)

        # compute attention from context
        # B x M (dest) x (M) (src)
        attn_cont = torch.matmul(query, key.transpose(-1, -2))

        # attention probabilities
        #attn_cont = _unskew(attn_cont)  # B x M x L

        # compute the effect of position embedding
        attn_pos = torch.matmul(query, key_pe)  # B x M x L_pos
        attn = attn_cont + attn_pos

        attn = attn / math.sqrt(self.hidden_size)  # B x M X L_pos
        attn = F.softmax(attn, dim=-1)

        if self.adapt_span_enabled:
            # trim attention lengths according to the learned span
            attn = self.adaptive_span(attn)
        attn = self.dropout(attn)  # B x M X L_pos

        #attn_cont = _skew(attn, 0)  # B x M X (L+M)
        out = torch.matmul(attn_cont, value)  # B x M x H

        return out

    def get_cache_size(self):
        if self.adapt_span_enabled:
            return self.adaptive_span.get_cache_size()
        else:
            return self.attn_span

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
                 attn_span, rev_net, **kargs):
        nn.Module.__init__(self)
        # token embeddings
        self.in_emb = nn.Embedding(vocab_size, hidden_size)
        self.out_emb = nn.Linear(hidden_size, vocab_size)
        # position embeddings
        self.key_pe = nn.Parameter(
            torch.randn(1, hidden_size // nb_heads, attn_span))
        # reversible blocks
        rev_block = (lambda x: x if not rev_net else
                     ReversibleBlock(f_block=x,
                                     g_block=SideRevNet(hidden_size=hidden_size),
                                     split_along_dim=-1))
        # transformer layers
        self.layers = nn.ModuleList([
            rev_block(TransformerLayer(hidden_size=hidden_size,
                                       nb_heads=nb_heads,                
                                       attn_span=attn_span, **kargs))
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
