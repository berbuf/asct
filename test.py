import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_span import AdaptiveSpan
from models import _skew, _unskew

# Reecrire en s'inspirant de adaptive

def calc_time(fct, name):
    torch.cuda.synchronize()
    start = time.time()
    fct()
    torch.cuda.synchronize()
    end = time.time()
    print(name, end - start)
    print ()

def get_idx(span, idx_template, align):
    _,M,_=span.size()
    left_bound = (torch.max(span[:,:,0], -span[:,:,1])
                                .unsqueeze(2).round().long())
    right_bound = (torch.max(-span[:,:,0], span[:,:,1])
                                  .unsqueeze(2).round().long())
    len_span = (left_bound + right_bound).abs() + 1 # central token
    max_span = len_span.max()

    # prepare idx_span
    idx_span = idx_template[:,:,:max_span]
    idx_span[:,:,] = align
    # align right, negative values are out of windows
    aligned_right = idx_span - (max_span - len_span)
    # scale to real values, and align left
    idx_span = (aligned_right + align - left_bound).flatten()
    # flatten for speed up, assign M to out of window indexes
    idx_span[(aligned_right.flatten() < 0) | (idx_span > M)] = M
    idx_span = idx_span.view(-1, max_span)

    return max_span, idx_span

def _slice(key, key_pe, span, idx_template, align):
    B,M,H=key.size()

    # idx_slices
    max_span, idx_span = get_idx(span, idx_template, align)
    
    # key
    key = F.pad(key, (0, 0, 0, 1))
    key = (key.view(-1, H)[idx_span]
           .view(B, M, max_span, H)
           .transpose(-1, -2)
           .view(B * M, H, max_span))

    # key_pe
    key_pe = key_pe.repeat(B, 1, 1)#.transpose(-1, -2)
    key_pe = F.pad(key_pe, (0, 0, 0, 1))
    key_pe = (key_pe.view(-1, H)[idx_span]
              .view(B, M, max_span, H)
              .transpose(-1, -2)
              .view(B * M, H, max_span))

    return key, key_pe

def scaling(query, key, value, key_pe, span, idx_template, align):
    B,M,H=query.size()

    # slices
    key, key_pe = _slice(key, key_pe, span, idx_template, align)

    query = query.view(B*M, 1, H)

    attn_cont = torch.bmm(query, key).view(B, M, -1)
    attn_pos = torch.bmm(query, key_pe).view(B, M, -1)
    attn = attn_cont + attn_pos

    attn = attn / math.sqrt(H)
    attn = F.softmax(attn, dim=-1)

    out = torch.mul(attn.unsqueeze(3), value.unsqueeze(2)).sum(2)

    return out

def adaptive(query, key, value, key_pe, adaptive_span):
    B,M,H=query.size()

    x, y, z = adaptive_span.trim_memory(query, key, value, key_pe)

    attn_cont = torch.matmul(query, key.transpose(-1, -2))
    # useless without cache
    a = _unskew(attn_cont)

    attn_pos = torch.matmul(query, key_pe.transpose(-1, -2))
    attn = attn_cont + attn_pos

    attn = attn / math.sqrt(H)
    attn = F.softmax(attn, dim=-1)

    t = adaptive_span(attn)

    # useless without cache
    b = _skew(attn, 0)
    out = torch.matmul(attn, value)
    return a, b, x, y, z, t, out
    
def scale(key, key_pe, span):
    B,M,H=key.size()

    left_bound = span[:,:,0].max().round().long()
    right_bound = span[:,:,1].max().round().long()

    max_left = left_bound.item()
    max_right = right_bound.item()
    max_span = max_left + 1 + max_right
    
    key = F.pad(key, (0, 0, max_left, max_right))
    key_pe = F.pad(key_pe, (0, 0, max_left, max_right))

    return key, key_pe, max_span

def scaling2(query, key, value, key_pe, span):
    B,M,H=query.size()

    # key = B X M X L X H
    key, key_pe, max_span = scale(key, key_pe, span)
    query = query.unsqueeze(2)
    key = key.transpose(-1, -2)
    a = torch.stack([ torch.matmul(query[:,N], key[:,:,N:max_span+N]) for N in range(M) ])

    # B X M X H * B X L X H

    return a

class SelectAttention(nn.Module):

    def __init__(self, batch_size, n_heads, max_span):
        nn.Module.__init__(self)
        B, K, M = batch_size, n_heads, max_span

        self.n_heads = K

        mask = (torch.linspace(0, M - 1, M)
                .repeat(B * M, 1)
                .reshape(B // K, K, M, M)
                - torch.arange(M).float().unsqueeze(1))
        self.register_buffer('mask', mask)

    def forward(self, attn, span):
        (B,M,_), K = attn.size(), self.n_heads

        attn = attn.reshape(B//K, K, M, -1)
        span = span.reshape(B//K, K, M, 2, 1)

        mean = span[:,:,:,1]
        intercept = span[:,:,:,0]

        # -(x+b)**2+a
        z = -(self.mask - intercept)**2 + mean
        z = z.clamp(0, 1)

        attn = attn * z

        return attn.reshape(B,M,M)
        
def scaling3(query, key, value, key_pe, span, select):
    B,M,H=query.size()
    attn_cont = torch.matmul(query, key.transpose(-1, -2))
    attn_pos = torch.matmul(query, key_pe)
    attn = attn_cont + attn_pos
    attn = select(attn, span)
    attn = attn / math.sqrt(H)
    attn = F.softmax(attn, dim=-1)
    # add dropout here
    out = torch.matmul(attn, value)
    return out

def main():
    #B,M,H=64,256,1024
    B,K,M,H=8,2,16,128
    limit_span=50

    torch.manual_seed(42)

    adaptive_span = AdaptiveSpan(attn_span=M, adapt_span_loss=0, adapt_span_ramp=32,
                                 adapt_span_init=0, adapt_span_cache=False, nb_heads=1)
    query = torch.FloatTensor(B*M*H).uniform_(0, 1).reshape(B, M, H)#.cuda()
    key = torch.FloatTensor(B*M*H).uniform_(0, 1).reshape(B, M, H)#.cuda()
    value = torch.FloatTensor(B*M*H).uniform_(0, 1).reshape(B, M, H)#.cuda()
    key_pe = torch.FloatTensor(M*H).uniform_(0, 1).reshape(1, H, M)#.cuda()
    # B_K, M, 2
    span = (torch.FloatTensor(B*M*2)
            .uniform_(0, 1)
            .reshape(B, M, 2)*10 -5)#.cuda() * 10 - 5

    align = torch.arange(M).unsqueeze(1)#.cuda()
    # basic span index of max_span size
    idx_template = (torch.arange(limit_span)
                    # one idx for each token
                    .unsqueeze(1).repeat(B * M, 1)
                    # format to rows
                    .transpose(0, 1).view(B, M, limit_span)
                    #.cuda()
                    + align)

    select = SelectAttention(B, K, M)
    #adaptive(query, key, value, key_pe, adaptive_span)
    #scaling2(query, key, value, key_pe, span)
    scaling3(query, key, value, key_pe, span, select)
    return

    # test
    for _ in range(100):
        calc_time(lambda: adaptive(query, key, value, key_pe, adaptive_span), "adaptive")
        calc_time(lambda: scaling(query, key, value, key_pe, span, idx_template, align), "scaling")
        calc_time(lambda: scaling2(query, key, value, key_pe, span), "scaling2")

main()
