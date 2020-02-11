import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from asa import AutoSelectAttention

# check cuda time
def calc_time(fct, name):
    torch.cuda.synchronize()
    start = time.time()
    fct()
    torch.cuda.synchronize()
    end = time.time()
    print(name, end - start)
    print ()

##
### SCALE WITH IDX
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
    key_pe = key_pe.repeat(B, 1, 1).transpose(-1, -2)
    key_pe = F.pad(key_pe, (0, 0, 0, 1))
    key_pe = (key_pe.view(-1, H)[idx_span]
              .view(B, M, max_span, H)
              .transpose(-1, -2)
              .view(B * M, H, max_span))

    return key, key_pe

def scale_idx(query, key, value, key_pe, span, idx_template, align):
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

##
### ORIGINAL ADAPTIVE
def adaptive(query, key, value, key_pe, adaptive_span):
    B,M,H=query.size()

    x, y, z = adaptive_span.trim_memory(query, key, value, key_pe)

    attn_cont = torch.matmul(query, key.transpose(-1, -2))
    # useless without cache
    a = _unskew(attn_cont)

    attn_pos = torch.matmul(query, key_pe)
    attn = attn_cont + attn_pos

    attn = attn / math.sqrt(H)
    attn = F.softmax(attn, dim=-1)

    t = adaptive_span(attn)

    # useless without cache
    b = _skew(attn, 0)
    out = torch.matmul(attn, value)
    return a, b, x, y, z, t, out

##
### SCALE WITH MASK
"""
class SelectAttention(nn.Module):

    def __init__(self, batch_size, n_heads, max_span):
        nn.Module.__init__(self)
        B, K, M = batch_size, n_heads, max_span

        self.n_heads = K

        mask = (torch.linspace(0, M - 1, M)
                .repeat(B * M, 1)
                .reshape(B // K, K, M, M)
                - torch.arange(M).float().unsqueeze(1)).cuda()
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
"""
        
def scale_mask(query, key, value, key_pe, span, select):
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
    #B,K,M,H=64,2,256,1024
    #B,K,M,H=8,2,16,128
    B,K,M,H=2,2,5,3
    #limit_span=50

    torch.manual_seed(42)

    #adaptive_span = AdaptiveSpan(attn_span=M, adapt_span_loss=0, adapt_span_ramp=32,
    #                             adapt_span_init=0, adapt_span_cache=False, nb_heads=1)
    query = torch.FloatTensor(B*M*H).uniform_(0, 1).reshape(B, M, H).cuda()
    key = torch.FloatTensor(B*M*H).uniform_(0, 1).reshape(B, M, H).cuda()
    value = torch.FloatTensor(B*M*H).uniform_(0, 1).reshape(B, M, H).cuda()
    key_pe = torch.FloatTensor(M*H).uniform_(0, 1).reshape(1, H, M).cuda()

    # B_K, M, 2
    span = (torch.FloatTensor(B*M*2)
            .uniform_(0, 1)
            .reshape(B, M, 2).cuda() * 10 - 5)

    #align = torch.arange(M).unsqueeze(1).cuda()
    # basic span index of max_span size
    #idx_template = (torch.arange(limit_span)
                    # one idx for each token
    #                .unsqueeze(1).repeat(B * M, 1)
                    # format to rows
    #                .transpose(0, 1).view(B, M, limit_span)
    #                .cuda()
    #                + align)

    select = AutoSelectAttention(B, K, M, 1.5)

    #adaptive(query, key, value, key_pe, adaptive_span)
    #scale_idx(query, key, value, key_pe, span, idx_template, align)
    scale_mask(query, key, value, key_pe, span, select)
    return

    # test
    for _ in range(100):
        calc_time(lambda: adaptive(query, key, value, key_pe, adaptive_span), "adaptive")
        calc_time(lambda: scale_idx(query, key, value, key_pe, span, idx_template, align), "scaling")
        calc_time(lambda: scale_mask(query, key, value, key_pe, span, select), "scaling3")

def res(f):
    print (f)
    l = [ e.split(" ") for e in open(f).readlines() if len(e) > 1]
    
    res = {}
    for name, time in l:
        if name not in res:
            res[name] = []
            res[name] += [float(time)]

    for n in res:
        print (n, sum(res[n]) / len(res[n]))

main()
#res("./res")
