import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import get_r2, round_r2

class AutoSelectAttention(nn.Module):
    """ dynamically select a span of attention for each token """

    def __init__(self, M):
        nn.Module.__init__(self)
        self.x = (torch.arange(M) - M//2).float()#.cuda()
        self.r2 = get_r2(M).float()#.cuda()
        self.M = M

    def set_pad_h(self, pad_h):
        self.pad_h = pad_h # BK x 1 x D

    def repeat_val(self, val, attn):
        """
        skew and reshape to: (B,M/L,3L,H), for parallel computation
        example: [1, 2, 3, 4, 5, 6] => [[0, 0, 1, 2, 3, 4]
                                        [1, 2, 3, 4, 5, 6]
                                        [3, 4, 5, 6, 0, 0]]
        """
        B,M,H=val.size()
        _,_,L,_=attn.size()
        pad_h = self.pad_h.repeat(1, L, 1) # repeat pad_h
        val = torch.cat((pad_h, val, pad_h), dim=1) # PAD vector
        val = val.reshape(B, M//L+2, 1, L, H) # Split by L blocks
        val = val.repeat(1, 1, 3, 1, 1) # Repeat blocks 3 times
        val = torch.cat((val[:,range(M//L),2],
                         val[:,range(1, M//L+1),1],
                         val[:,range(2, M//L+2),0]),
                        dim=2) # Cat in diagonal, # B x S x 3L x H
        return val

    def get_maximum_direction(self, M, mean, variance):
        """
        return maximum span in left or right direction
        round to a power of two
        """
        L = (mean.abs() + variance.abs()).max().ceil()#.int() # pdf > 80%
        L = round_r2(self.r2, L*2) # pdf > 99%
        if L * 4 > M: # don't over expand
            L = M // 4
        return L

    def forward(self, span):
        """
        attention is defined as y = -((x+a)/b)**2
        compute attention over a block of 2 maximum span (4L)
        skew to shape: (B,M/L,L,3L), for parallel computation
        example: [[-4,-3,-2,-1,0,1,2,3], => [[-2,-1, 0,1,2,3]
                  [-4,-3,-2,-1,0,1,2,3]]     [-3,-2,-1,0,1,2]]
        """
        B,M,_=span.size()
        mean, variance = (span[:,:,0].unsqueeze(2),
                          span[:,:,1].unsqueeze(2))
        L = self.get_maximum_direction(M, mean, variance)
        x = self.x[self.M//2 - L*2 : self.M//2 + L*2] # 4L
        y = -((x + mean) / (variance + 1e-5))**2 # B x M x 4L
        #y = x.reshape(1, 1, len(x)).repeat(y.shape[0], y.shape[1], 1)
        S = M // L # Number of Blocks
        y = y.reshape(B, S, -1) # B x S x L(4L)
        y = y[:,:,1:-L+1] # B x S x L(4L) - L
        y = y.reshape(B, S, L, -1) # B x S x L x 4L - 1
        y = y[:,:,:,(L-1):] # B x S x L x 3L
        return y
