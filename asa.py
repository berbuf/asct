import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AutoSelectAttention(nn.Module):
    """ dynamically select a span of attention for each token """

    def __init__(self, M):
        nn.Module.__init__(self)
        #self.x = (torch.arange(L * 4) - L * 2 + 1).cuda()
        self.x = (torch.arange(M) - M//2).cuda()
        self.m2, e = [], 0
        while 2**e < M:
            e += 1
            self.m2 += [2**e] # sorted power of 2
        self.m2 = torch.tensor(self.m2).cuda()
        self.M = M

    def set_pad_h(self, pad_h):
        self.pad_h = pad_h # BK x 1 x D

    def repeat_val(self, val):
        B,M,H=val.size()
        L = self.L
        pad_h = self.pad_h.repeat(1, L, 1) # repeat pad_h
        val = torch.cat((pad_h, val, pad_h), dim=1) # PAD vector
        val = val.reshape(B, M//L+2, 1, L, H) # Split by L blocks
        val = val.repeat(1, 1, 3, 1, 1) # Repeat blocks 3 times
        val = torch.cat((val[:,range(M//L),2],
                         val[:,range(1, M//L+1),1],
                         val[:,range(2, M//L+2),0]),
                        dim=2) # Cat in diagonal
        return val # B x S x 3L x H  

    def forward(self, span):
        B,M,_=span.size()
        mean, softness = (span[:,:,0].unsqueeze(2),
                          span[:,:,1].unsqueeze(2))
        self.L = 64
        #self.L = (mean.abs() + softness.abs()).max().ceil() # pdf > 80%
        #self.L = self.m2[(self.m2 < self.L).sum()] # round to power of 2
        x = self.x[self.M//2-self.L*2:self.M//2+self.L*2] # 4L, pdf > 99%
        y = -((x + mean) / (softness + 1e-5))**2 # B x M x 4L
        S = M // self.L #S = Number of Blocks
        print (M, S)
        y = y.reshape(B, S, -1) # B x S x L(4L)
        y = y[:,:,:-self.L] # B x S x L(4L) - L
        y = y.reshape(B, S, self.L, -1) # B x S x L x 4L - 1
        y = y[:,:,:,(self.L-1):] # B x S x L x 3L
        return y
