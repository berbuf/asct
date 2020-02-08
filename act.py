#!/usr/bin/env python3

import torch
import torch.nn as nn

class AdaptiveComputationTime(nn.Module):

    def __init__(self, block_size, hidden_size, threshold,
                 dup_batch_size, **kargs):
        nn.Module.__init__(self)

        self.p = nn.Linear(hidden_size, 1)
        self.sigma = nn.Sigmoid()
        self.threshold = threshold
        # accumulated p probability
        self.acc_p = torch.zeros(dup_batch_size, block_size, 1).cuda()
        # remaining probability
        self.remainder = torch.zeros(dup_batch_size, block_size, 1).cuda()
        # store_weights
        self.weights = torch.zeros(dup_batch_size, block_size, hidden_size).cuda()
        # nb updates
        self.updates = 0

    def forward(self, h):
        """ adaptive computation time mechanism"""
        # halt probability
        p = self.sigma(self.p(h))
        # masks
        mask_run = self.acc_p < 1.0
        mask_exit = (self.acc_p + p) * mask_run >= self.threshold
        mask_continue = (mask_exit ^ 1) * mask_run
        # store halt and remainder probability
        self.acc_p = self.acc_p + p * mask_run
        self.remainder = self.remainder + (1 - p) * mask_exit
        update = (
            # current p if continue
            p * mask_continue +
            # remainder if exits
            (1 - p) * mask_exit
            # 0 if pruned
        )
        # update the share of h weights
        self.weights = (
            # current p or remainder
            h * update +
            # downweight prev weights
            # or keep unchanged if pruned
            self.weights * (1 - update)
        )
        # store nb of steps
        self.updates += 1
        # True if all tokens have exited 
        return (self.acc_p >= self.threshold).byte().any()

    def loss(self):
        """ minimize number of updates and remaining probability """
        return self.updates, self.remainder
