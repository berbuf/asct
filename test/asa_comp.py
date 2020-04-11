import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoSelectAttention(nn.Module):
    """ dynamically select a span of attention for each token """

    def __init__(self, batch_size, n_heads, block_size, L):
        nn.Module.__init__(self)
        self.max_block = L
        self.L = L
        # skew mask
        self.x = (torch.arange(self.max_block*2-1) -
                  self.max_block + 1).cuda()
        self.x_ = (torch.arange(block_size) -
                   torch.arange(block_size)
                   .unsqueeze(1)).cuda()
        self.x = self.x.repeat(batch_size * n_heads, 1, 1)
        self.x_2 = (torch.arange(L * 4) - L * 2 + 1).cuda() 

    def _skew(self, X, pad_value):
        B, M, H = X.size()
        X = F.pad(X, (0, M + 1), value=pad_value)
        X = X.view(B, -1)
        X = X[:, :-M]
        X = X.view(B, M, M + H)
        return X

    def forward2(self, span):
        B,M,_=span.size()
        mean, softness, intercept = (span[:,:,0].unsqueeze(2),
                                     span[:,:,1].unsqueeze(2),
                                     span[:,:,2].unsqueeze(2))
        S = M // self.L #S = Number of Blocks
        y = -((self.x_2 + mean) / (softness + 1e-5))**2 + intercept # B x M x 4L
        y = y.reshape(B, S, -1) # B x S x L(4L)
        y = y[:,:,:-self.L] # B x S x L(4L) - L
        y = y.reshape(B, S, self.L, -1) # B x S x L x 4L - 1
        y = y[:,:,:,(self.L-1):] # B x S x L x 3L
        y = F.softmax(y, dim=-1)
        return y#.contiguous()

    def forward(self, span, noskew=False):
        M = self.max_block
        # isolate variables
        mean = span[:,:,0].unsqueeze(2)
        softness = span[:,:,1].unsqueeze(2)
        intercept = span[:,:,2].unsqueeze(2)
        if noskew:
            y = -((self.x_ + mean) / (softness + 1e-5))**2 + intercept
            y = F.softmax(y, dim=-1)
            return y

        # select function
        # y = -((x+a)/soft)**2+b
        y = -((self.x + mean) / (softness + 1e-5))**2 + intercept
        return y
        # skew
        #y = F.softmax(y, dim=-1) 
        #y = self._skew(y, 0)
        # crop
        #y = y[:,:,M-1:-M]
        #return y
