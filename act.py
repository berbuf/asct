#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveComputationTime(nn.Module):

    def __init__(self, block_size, hidden_size, threshold,
                 dup_batch_size, **kargs):
        nn.Module.__init__(self)

        self.p = nn.Linear(hidden_size, 1)
        self.sigma = nn.Sigmoid()
        self.threshold = threshold
        self.mask_run = torch.ones(dup_batch_size, block_size, 1).bool().cuda()
        self.prev_continue = torch.ones(dup_batch_size, block_size, 1).bool().cuda()
        # accumulated p probability
        self.acc_p = torch.zeros(dup_batch_size, block_size, 1).cuda()
        # remaining probability
        self.remainders = torch.zeros(dup_batch_size, block_size, 1).cuda()
        # store_weights
        self.weights = torch.zeros(dup_batch_size, block_size, hidden_size).cuda()
        # nb updates
        self.updates = 0

    def prune_tokens(self, h, mask):
        B,M,H = h.size()
        sum_ = mask.sum(1)
        max_ = sum_.max()
        pad = max_ - sum_.min()
        arange_pad = (torch.arange(pad) + 1 -
                      (sum_ - max_ + pad).unsqueeze(1))        
        add_mask = arange_pad.clamp(0, 1).byte()
        mask = torch.cat((mask, add_mask), 1)
        h = f.pad(h, (0, 0, 0, pad))
        return h[mask].reshape(B, -1, H).contiguous()


    def forward(self, h):
        """ adaptive computation time mechanism"""        
        # halt probability
        p = self.sigma(self.p(h)).reshape(-1) # B_(M - E)
        # masks
        acc_p = self.acc_p[self.mask_run]
        mask_exit = (acc_p + p >= self.threshold).bool()
        mask_continue = (~mask_exit).bool()
        # update containers
        self.acc_p[self.mask_run] += p
        self.remainders[self.mask_run] += (1 - p) * mask_exit
        update = (
            # current p if continue
            p * mask_continue +
            # remainder if exits
            (1 - p) * mask_exit
        ).unsqueeze(1)
        # recompute manually
        # h has padding token !!!
        # update only running weights
        w = self.weights.unsqueeze(2)[self.mask_run] # B_(M - E) X H
        self.weights.unsqueeze(2)[self.mask_run] = (
            # current p or remainder
            h[self.prev_continue] * update +
            #h.view(-1, h.size(-1)) * update +
            # downweight prev weights
            w * (1 - update)
        )

        self.mask_run = self.acc_p >= 1.

        # store nb of steps
        self.updates += 1

        # prune tokens for speed up
        h = self.prune_tokens(h, mask_continue)
        self.prev_continue = mask_continue
        
        return h

    def loss(self):
        """ minimize number of updates and remaining probability """
        return self.updates, self.remainder
