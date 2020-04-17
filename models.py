#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from asa import AutoSelectAttention
from act import AdaptiveComputationTime

from torch.nn import CrossEntropyLoss, MSELoss

# Size notations:
# B = batch_size, H = hidden_size, M = block_size, K = nb_heads

class SelfAttention(nn.Module):
    """ self-attention with selective attention """

    def __init__(self, hidden_size, dropout,
                 nb_heads, block_size, **kargs):
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.select = AutoSelectAttention(M=block_size)

    def forward(self, span, value):
        """ light attention implementation """ 
        B,M,H=value.size()
        # select attention
        attn = self.select(span)
        attn = F.softmax(attn, dim=-1)
        #attn = self.dropout(attn)
        # project to inner dim
        value = self.select.repeat_val(value, attn)
        #print (attn.shape)
        #print (value.shape)
        out = torch.matmul(attn, value)
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
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_span = nn.Linear(hidden_size, 2*nb_heads, bias=False)
        # loss term span
        self.norm_span = 0.

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

        span = self.head_reshape(self.proj_span(h), 2)
        value = self.head_reshape(self.proj_val(h),
                                  self.head_dim)

        out = self.attn(span, value)  # B_K x M x D
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)

        # span norm
        #self.norm_span += span.norm().mean()

        return out

# Boom layer
class FeedForwardLayer(nn.Module):

    def __init__(self, hidden_size, inner_hidden_size,
                 dropout, **kargs):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(hidden_size, inner_hidden_size, bias=True)
        self.fc2 = nn.Linear(inner_hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h1 = F.relu(self.fc1(h))
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2

class TransformerLayer(nn.Module):

    def __init__(self, batch_size, block_size, hidden_size,
                 nb_heads, **kargs):
        nn.Module.__init__(self)
        self.attn = MultiHeadSelfAttention(
            block_size=block_size,
            hidden_size=hidden_size,
            nb_heads=nb_heads,
            **kargs)
        self.ff = FeedForwardLayer(hidden_size=hidden_size,
                                   **kargs)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, h):
        # h = B x M x H
        attn_out = self.attn(h)
        h = self.norm1(h + attn_out)  # B x M x H
        ff_out = self.ff(h)
        out = self.norm2(h + ff_out)  # B x M x H
        return out

class Generator(nn.Module):
    def __init__(self, vocab_size, batch_size, hidden_size,
                 nb_heads, nb_layers_gen, block_size, **kargs):
        nn.Module.__init__(self)
        # decoder
        self.out_emb = nn.Linear(hidden_size, vocab_size, bias=True)
        # transformer layers
        self.layer = TransformerLayer(batch_size=batch_size,
                                      block_size=block_size,
                                      hidden_size=hidden_size,
                                      nb_heads=nb_heads,
                                      **kargs)
        self.nb_layers = nb_layers_gen

    def forward(self, h):
        for _ in range(self.nb_layers):
            h = self.layer(h)  # B x M x H
        # decoder
        h = self.out_emb(h)
        out = F.log_softmax(h, dim=-1)
        return out

class Discriminator(nn.Module):

    def __init__(self, vocab_size, batch_size, hidden_size,
                 nb_heads, nb_layers_disc, block_size, **kargs):
        nn.Module.__init__(self)
        # decoder
        #self.out_emb = nn.Linear(hidden_size, block_size, bias=False)
        self.out_emb = nn.Linear(hidden_size, 1, bias=True)
        # transformer layers
        self.layer = TransformerLayer(batch_size=batch_size,
                                      block_size=block_size,
                                      hidden_size=hidden_size,
                                      nb_heads=nb_heads,
                                      **kargs)
        self.nb_layers = nb_layers_disc
        if self.nb_layers != -1:
            self.act_module = AdaptiveComputationTime(
                batch_size, block_size,
                hidden_size, **kargs)
            self.forward = self.forward_act
        else:
            self.forward = self.forward_no_act

    def forward_no__act(self, h, pad_h):
        for _ in range(self.nb_layers):
            h = self.layer(h)  # B x M x H
        # decoder
        h = self.out_emb(h)
        out = torch.sigmoid(h)
        return h, out

    def forward_act(self, h, pad_h):
        # init act
        self.act_module.init_act(pad_h)
        self.layer.attn.norm_span = 0.

        # loop until empty
        _,M,_=h.size()
        while M:
            h = self.layer(h)  # B x M x H
            h = self.act_module(h)
            _,M,_=h.size()
            #print (M)
        h = self.act_module.weighted_h
        # decoder
        out = torch.sigmoid(self.out_emb(h))
        return h, out

class GenDisc(nn.Module):

    def __init__(self, vocab_size, batch_size, model_params, pad_idx):
        nn.Module.__init__(self)
        # Shared token embeddings
        self.in_emb = nn.Embedding(vocab_size,
                                   model_params["hidden_size"])
        self.gen = Generator(vocab_size, batch_size,
                             **model_params)
        self.disc = Discriminator(vocab_size, batch_size,
                                  **model_params)
        self.pad_idx = torch.tensor([pad_idx]).cuda()

    def forward(self, x_masked):
        h = self.in_emb(x_masked)
        # pad vector
        pad_h = self.in_emb(self.pad_idx)[0]
        # log p output of generator
        out_gen = self.gen(h)
        # generate tokens
        x_gen = out_gen.argmax(2)
        h = self.in_emb(x_gen)
        # discriminate
        _, out_disc = self.disc(h, pad_h)
        return out_gen, out_disc

class AsctSequenceClassification(nn.Module):
    def __init__(self, task_config, model_params, asct, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.in_emb = asct.in_emb
        self.disc = asct.disc
        self.dropout = nn.Dropout(model_params["dropout"])
        self.pad_idx = asct.pad_idx
        # pooler
        self.dense = nn.Linear(model_params["hidden_size"],
                               model_params["hidden_size"], bias=True)
        self.act = nn.ReLU()
        # classifier
        self.cls = nn.Linear(model_params["hidden_size"], self.num_labels, bias=True)

    def forward(self, input_ids, labels):
        # embeds
        h = self.in_emb(input_ids)
        # pad vector
        pad_h = self.in_emb(self.pad_idx)[0]
        # features
        h, _ = self.disc(h, pad_h)
        # pooler
        h_cls = h[:, 0]
        h_cls = self.dense(h_cls)
        h_cls = self.act(h_cls)
        # classifier
        h_cls = self.dropout(h_cls)
        logits = self.cls(h_cls)

        if self.num_labels == 1:
            #  Regression
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            # Classification
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits
