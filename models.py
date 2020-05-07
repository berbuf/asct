#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from asa import AutoSelectAttention
from act import AdaptiveComputationTime

from vanilla_model import Vanilla

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
        attn = self.dropout(attn)
        # project to inner dim
        value = self.select.repeat_val(value, attn)
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
        self.norm_span = None

    def set_pad_h(self, pad_h, B):
        # set pad vector for asa
        pad_h = pad_h.repeat(B, 1, 1) # B x 1 x H
        pad_h = self.head_reshape(pad_h, self.head_dim) # BK x 1 x D
        self.attn.select.set_pad_h(pad_h)

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
        #value = self.head_reshape(h, self.head_dim)
        value = self.head_reshape(self.proj_val(h),
                                  self.head_dim)
        out = self.attn(span, value)  # B_K x M x D
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        self.norm_span += torch.abs(span).mean() # l1 penalty loss
        self.tmp_span = span
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
        #return attn_out
        h = self.norm1(h + attn_out)  # B x M x H
        ff_out = self.ff(h)
        out = self.norm2(h + ff_out)  # B x M x H
        return out

class Asct(nn.Module):
    def __init__(self, batch_size, hidden_size,
                 nb_heads, block_size, nb_layers, **kargs):
        nn.Module.__init__(self)
        self.layer = TransformerLayer(batch_size=batch_size,
                                      block_size=block_size,
                                      hidden_size=hidden_size,
                                      nb_heads=nb_heads,
                                      **kargs)
        self.act = AdaptiveComputationTime(batch_size, block_size,
                                           hidden_size, **kargs)
        self.max_layers = nb_layers

    def forward(self, h, pad_h):
        self.track_act = []
        self.track_asa = []

        B,M,_=h.size()
        self.act.init_act(pad_h) # init act
        self.layer.attn.set_pad_h(pad_h, B) # set pad h for asa
        if self.layer.attn.norm_span != None:
            self.layer.attn.norm_span.detach_()
        self.layer.attn.norm_span = 0. # reset norm span
        max_layers = self.max_layers
        while M: # loop until empty
            h = self.layer(h)

            # track values
            L = self.layer.attn.attn.select.track_L
            if type(L) != int:
                L = L.item()
            self.track_asa += [L]
            self.track_act += [self.act.run.float().sum(1).mean().item()]

            h = self.act(h)
            _,M,_=h.size()

            max_layers -= 1
            if not max_layers: # added max layers limit
                break

        h = self.act.weighted_h
        return h

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
        h = self.out_emb(h) # decoder
        out = F.log_softmax(h, dim=-1)
        return out

class GenDisc(nn.Module):
    def __init__(self, vocab_size, batch_size, model_params, pad_idx):
        nn.Module.__init__(self)
        self.in_emb = nn.Embedding(vocab_size,
                                   model_params["hidden_size"]) # Shared token embeddings
        self.gen = Generator(vocab_size, batch_size,
                             **model_params)
        self.disc = Asct(vocab_size, batch_size,
                         **model_params)
        self.pad_idx = torch.tensor([pad_idx]).cuda()

    def forward(self, x_masked):
        h = self.in_emb(x_masked)
        pad_h = self.in_emb(self.pad_idx)[0] # pad vector
        out_gen = self.gen(h) # log p output of generator
        x_gen = out_gen.argmax(2) # generate tokens
        h = self.in_emb(x_gen)
        _, out_disc = self.disc(h, pad_h) # discriminate
        return out_gen, out_disc

class DecoderClassification(nn.Module):
    def __init__(self, model_params, num_labels, pad_idx):
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(model_params["dropout"])
        self.num_labels = num_labels
        self.dense = nn.Linear(model_params["hidden_size"],
                               model_params["hidden_size"], bias=True)
        self.act = nn.ReLU()
        self.cls = nn.Linear(model_params["hidden_size"], self.num_labels, bias=True)
        self.vanilla = model_params["vanilla"]
        self.pad_idx = pad_idx

    def forward(self, h, y, x):
        if not self.vanilla:
            # take average of vectors inside a span
            t = []
            for b_h, b_y in zip(h, y):
                for s in b_y:
                    if s[2] == -1:
                        break
                    t += [b_h[s[0]:s[1]].mean(0)]
            h = torch.stack(t)
            y = y.view(-1, 3)
            y = y[y[:,2] != -1][:,2]
        else:
            t = []
            for i in range(len(y)):
                t += [h[i][x[i] != self.pad_idx].mean(0)]
            h = torch.stack(t)

        h = self.dropout(h)
        h = self.dense(h)
        h = self.act(h)
        h = self.cls(h)
        return h, y

class AsctImdbClassification(nn.Module):
    def __init__(self, batch_size, weights_embed, pad_idx, model_params, vanilla):
        nn.Module.__init__(self)
        vocab_size, emb_hidden_size = weights_embed.size()
        self.in_emb = nn.Embedding(vocab_size, emb_hidden_size)
        self.in_emb.load_state_dict({'weight': weights_embed}) # load glove
        self.out_emb = nn.Linear(emb_hidden_size, model_params["hidden_size"], bias=True)
        if vanilla:
            self.model = Vanilla(batch_size, **model_params)
        else:
            self.model = Asct(batch_size, **model_params)
        self.decoder = DecoderClassification(model_params,
                                             num_labels=2, pad_idx=pad_idx)
        self.pad_idx = torch.tensor([pad_idx]).cuda()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.loss = torch.nn.NLLLoss()

    def forward(self, x, y):
        h = self.in_emb(x) # embed
        h = self.out_emb(h)

        pad_h = self.in_emb(self.pad_idx)[0] # pad vector
        pad_h = self.out_emb(pad_h)

        h = self.model(h, pad_h)
        
        out, y = self.decoder(h, y, x) # decoder

        out = self.log_softmax(out)

        loss = self.loss(out, y)

        acc = (out.argmax(1) == y).float().mean()
        return loss, acc
