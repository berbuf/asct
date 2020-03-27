import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoSelectAttention(nn.Module):
    """ dynamically select a span of attention for each token """

    def __init__(self, n_heads, block_size):
        nn.Module.__init__(self)
        # skew mask
        self.x = (torch.arange(block_size) -
                  torch.arange(block_size)
                  .unsqueeze(1)).cuda()

    def forward(self, span):
        _, M, _ = span.size()        
        # isolate variables
        mean = span[:,:,0].unsqueeze(2)
        intercept = span[:,:,1].unsqueeze(2)
        # select function
        # y = -((x+a)/soft)**2+b
        y = -((self.x[:M,:M] + mean) / self.soft)**2 + intercept
        y = y.clamp(0)
        # normalize
        y = y / y.sum(2)
        return y
