import time
import math
import torch
import torch.nn.functional as F
from asa_comp import AutoSelectAttention

def ram():
    t = torch.cuda.get_device_properties(device).total_memory
    c = torch.cuda.memory_cached(device)
    a = torch.cuda.memory_allocated(device)
    f = c-a  # free inside cache
    print (t, c, a, f)

def head_reshape(x, K, head_dim):
    D = head_dim
    x = x.view(x.size()[:-1] + (K, D))  # B x M x K x D
    x = x.transpose(1, 2).contiguous()  # B x K x M x D
    x = x.view(-1, x.size(-2), x.size(-1))  # B_K x M x D
    return x    

def duplicate_value(val, L):
    B, M, H = val.size()
    assert not (M % L)
    ML = L * 2 - 1
    S = M % ML
    val = val.repeat(1, ML, 1).view(B, ML, M, H)
    val = F.pad(val, (0, 0, L-1, L-1)) # B x ML x M + ML-1 x H
    val = (F.pad(val.view(B, -1, H), (0, 0, 0, ML)) # B x ML x M + ML x H
           .view(B, ML, -1, H))
    val = val[:,:,:-S] # B x ML x F ML  x H
    val = val.view(B, ML, -1, ML, H) # B x ML x F x ML  x H
    val = val.transpose(1, 2) # B x F x ML x ML x H
    val = val.contiguous().view(B, -1, ML, H)  # B x F ML x ML x H
    val = val[:,:M] # B x M x ML x H
    return val

def repeat_val(val, pad, L):
    B,M,H=val.size()
    val = torch.cat((pad, val, pad), dim=1) # PAD vector
    val = val.reshape(B, M//L+2, 1, L, H) # Split by L blocks
    val = val.repeat(1, 1, 3, 1, 1) # Repeat blocks 3 times
    val = torch.cat((val[:,range(M//L),2],
                     val[:,range(1, M//L+1),1],
                     val[:,range(2, M//L+2),0]),
                    dim=2) # Cat in diagonal
    return val#.contiguous() # B x S x 3L x H

B,M,H,K=1,2048,512,8
V=1000
D=H//K
L=32

assert (not M % L)

iter=1000

device=1
torch.cuda.set_device(device)

emb = torch.nn.Embedding(V, H).cuda()
proj_query = torch.nn.Linear(H, H, bias=False).cuda()
proj_key = torch.nn.Linear(H, H, bias=False).cuda()
proj_val = torch.nn.Linear(H, H, bias=False).cuda()
proj_span = torch.nn.Linear(H, 3*K, bias=False).cuda()
proj_out = torch.nn.Linear(H,H, bias=False).cuda()

proj_val_left = torch.nn.Linear(D, D, bias=False).cuda()
proj_val_mid = torch.nn.Linear(D, D, bias=False).cuda()
proj_val_right = torch.nn.Linear(D, D, bias=False).cuda()

proj_val_test = torch.nn.Linear(D, D*3, bias=False).cuda()

pad = torch.zeros((B*K, L, D)).cuda() # PAD vector

select = AutoSelectAttention(B, K, M, L)

X = torch.arange(B*M).reshape(B,M).cuda() % V

h = emb(X)

print ("here")

## ASA IDEAL
m000a = []
m000b = []
m000c = []
for _ in range(iter):

    # n^2
    start = time.time()
    span = head_reshape(proj_span(h), K, 3)
    attn = select.forward2(span)
    end = time.time()
    m000a += [end - start]

    # 3 val
    start = time.time()
    value = head_reshape(proj_val(h), K, D)
    value = proj_val_test(value).reshape(B*K, M//L, L*3, D)
    end = time.time()
    m000b += [end - start]

    # mult
    start = time.time()
    out = torch.matmul(attn, value).reshape(-1, M, D)
    end = time.time()
    m000c += [end - start]

    out = out.view(B, K, M, D)  # B x K x M x D
    out = out.transpose(1, 2).contiguous()  # B x M x K x D
    h = out.view(B, M, -1)  

    h = h.detach()

print ("asa ideal")
print (sum(m000a) / len(m000a))
print (sum(m000b) / len(m000b))
print (sum(m000c) / len(m000c))
print ((sum(m000a)+sum(m000b)+sum(m000c)) / len(m000a) )
print ()


## ASA VAL x 3
m00a = []
m00b = []
m00c = []
for _ in range(iter):
    # n^2
    start = time.time()
    span = head_reshape(proj_span(h), K, 3)
    attn = select.forward2(span)
    end = time.time()
    m00a += [end - start]

    # 3 val
    start = time.time()
    value = head_reshape(proj_val(h), K, D)
    value = torch.cat((pad, value, pad), dim=1) # PAD vector
    val_l = proj_val_left(value).reshape(B*K, M//L+2, L, D)
    val_m = proj_val_mid(value).reshape(B*K, M//L+2, L, D)
    val_r = proj_val_right(value).reshape(B*K, M//L+2, L, D) # B, S, L, H
    value = torch.cat((val_l[:,range(M//L)],
                       val_m[:,range(1, M//L+1)],
                       val_r[:,range(2, M//L+2)]),
                      dim=2) # Cat in diagonal
    end = time.time()
    m00b += [end - start]

    # mult
    start = time.time()
    out = torch.matmul(attn, value).reshape(-1, M, D)
    end = time.time()
    m00c += [end - start]

    out = out.view(B, K, M, D)  # B x K x M x D
    out = out.transpose(1, 2).contiguous()  # B x M x K x D
    h = out.view(B, M, -1)  

    h = h.detach()

print ("asa 3 val")
print (sum(m00a) / len(m00a))
print (sum(m00b) / len(m00b))
print (sum(m00c) / len(m00c))
print ((sum(m00a)+sum(m00b)+sum(m00c)) / len(m00a) )
print ()


## ASA DUP VAL
m0a = []
m0b = []
m0c = []
for _ in range(iter):

    # n^2
    start = time.time()
    span = head_reshape(proj_span(h), K, 3)
    attn = select.forward2(span)
    end = time.time()
    m0a += [end - start]

    # dup attn
    start = time.time()
    value = head_reshape(proj_val(h), K, D)
    value = repeat_val(value, pad, L)
    end = time.time()
    m0b += [end - start]

    # mult
    start = time.time()
    out = torch.matmul(attn, value).reshape(-1, M, D)
    end = time.time()
    m0c += [end - start]

    out = out.view(B, K, M, D)  # B x K x M x D
    out = out.transpose(1, 2).contiguous()  # B x M x K x D
    h = out.view(B, M, -1)  

    h = h.detach()

print ("asa dup val")
print (sum(m0a) / len(m0a))
print (sum(m0b) / len(m0b))
print (sum(m0c) / len(m0c))
print ((sum(m0a)+sum(m0b)+sum(m0c)) / len(m0a) )
print ()

## ASA
m2a = []
m2b = []
m2c = []
for _ in range(iter):

    # n^2
    start = time.time()
    span = head_reshape(proj_span(h), K, 3)
    attn = select(span)
    end = time.time()
    m2a += [end - start]

    # dup attn
    start = time.time()
    attn = select._skew(attn, 0)
    attn = F.softmax(attn, dim=-1)
    max = select.max_block
    attn = attn[:,:,max-1:-max]
    end = time.time()
    m2b += [end - start]

    # mult
    start = time.time()
    value = head_reshape(proj_val(h), K, D)
    out = torch.matmul(attn, value)
    end = time.time()
    m2c += [end - start]

    out = out.view(B, K, M, D)  # B x K x M x D
    out = out.transpose(1, 2).contiguous()  # B x M x K x D
    h = out.view(B, M, -1)  

    h = h.detach()

print ("asa")
print (sum(m2a) / len(m2a))
print (sum(m2b) / len(m2b))
print (sum(m2c) / len(m2c))
print ((sum(m2a)+sum(m2b)+sum(m2c)) / len(m2a) )
print ()

## ASA FULL
_m2a = []
_m2b = []
for _ in range(iter):

    # n^2
    start = time.time()
    span = head_reshape(proj_span(h), K, 3)
    attn = select(span, noskew=True)
    end = time.time()
    _m2a += [end - start]

    # mult
    start = time.time()
    value = head_reshape(proj_val(h), K, D)
    out = torch.matmul(attn, value)
    end = time.time()
    _m2b += [end - start]

    out = out.view(B, K, M, D)  # B x K x M x D
    out = out.transpose(1, 2).contiguous()  # B x M x K x D
    h = out.view(B, M, -1)  

    h = h.detach()

print ("asa full")
print (sum(_m2a) / len(_m2a))
print (sum(_m2b) / len(_m2b))
print ((sum(_m2a)+sum(_m2b)) / len(_m2a) )
print ()

"""
## ASA DUP
m2a_ = []
m2b_ = []
m2c_ = []
for _ in range(iter):

    # n^2
    start = time.time()
    span = head_reshape(proj_span(h), K, 3)
    attn = select(span)
    attn = F.softmax(attn, dim=-1)
    end = time.time()
    m2a_ += [end - start]

    # dup
    start = time.time()
    attn = attn.unsqueeze(2)
    value = head_reshape(proj_val(h), K, D)
    value = duplicate_value(value, L)
    end = time.time()
    m2b_ += [end - start]

    # mult
    start = time.time()
    out = torch.matmul(attn, value).squeeze()
    end = time.time()
    m2c_ += [end - start]

    out = out.view(B, K, M, D)  # B x K x M x D
    out = out.transpose(1, 2).contiguous()  # B x M x K x D
    h = out.view(B, M, -1)  

    h = h.detach()

print ("asa dup")
print (sum(m2a_) / len(m2a_))
print (sum(m2b_) / len(m2b_))
print (sum(m2c_) / len(m2c_))
print ((sum(m2a_)+sum(m2b_)+sum(m2c_)) / len(m2a_) )
print ()
"""

## Attention
ma = []
mb = []
mc = []
for _ in range(iter):

    # n^2
    start = time.time()
    query = head_reshape(proj_query(h), K, D)
    key = head_reshape(proj_key(h), K, D)
    attn = torch.matmul(query, key.transpose(-1, -2))
    end = time.time()
    ma += [end - start]

    # plus
    start = time.time()
    value = head_reshape(proj_val(h), K, D)
    attn = attn / math.sqrt(H)
    attn = F.softmax(attn, dim=-1)
    end = time.time()
    mb += [end - start]

    # mult
    start = time.time()
    out = torch.matmul(attn, value)
    end = time.time()
    mc += [end - start]

    out = out.view(B, K, M, D)  # B x K x M x D
    out = out.transpose(1, 2).contiguous()  # B x M x K x D
    h = out.view(B, M, -1)
    h = h.detach()

print ("attention")
print (sum(ma) / len(ma))
print (sum(mb) / len(mb))
print (sum(mc) / len(mc))
print ((sum(ma)+sum(mb)+sum(mc)) / len(ma) )
print ()
