import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoSelectAttention(nn.Module):
    """ dynamically select a span of attention for each token """

    def __init__(self, L):
        nn.Module.__init__(self)
        self.L = L
        self.x = (torch.arange(L * 4) - L * 2 + 1).cuda()

    def set_pad_h(self, pad_h):
        self.pad_h = pad_h # BK x L x D

    def repeat_val(self, val):
        B,M,H=val.size()
        L = self.L
        assert (not M % L)
        val = torch.cat((self.pad_h, val, self.pad_h), dim=1) # PAD vector
        val = val.reshape(B, M//L+2, 1, L, H) # Split by L blocks
        val = val.repeat(1, 1, 3, 1, 1) # Repeat blocks 3 times
        val = torch.cat((val[:,range(M//L),2],
                         val[:,range(1, M//L+1),1],
                         val[:,range(2, M//L+2),0]),
                        dim=2) # Cat in diagonal
        return val # B x S x 3L x H  

    def forward(self, span):
        B,M,_=span.size()
        mean, softness, intercept = (span[:,:,0].unsqueeze(2),
                                     span[:,:,1].unsqueeze(2),
                                     span[:,:,2].unsqueeze(2))
        S = M // self.L #S = Number of Blocks
        y = -((self.x + mean) / (softness + 1e-5))**2 + intercept # B x M x 4L
        y = y.reshape(B, S, -1) # B x S x L(4L)
        y = y[:,:,:-self.L] # B x S x L(4L) - L
        y = y.reshape(B, S, self.L, -1) # B x S x L x 4L - 1
        y = y[:,:,:,(self.L-1):] # B x S x L x 3L
        return y
