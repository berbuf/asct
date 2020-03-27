import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AdaptiveComputationTime(nn.Module):

    def __init__(self, batch_size, block_size,
                 hidden_size, threshold, **kargs):
        nn.Module.__init__(self)
        B,M,H=batch_size, block_size, hidden_size
        self.p = nn.Linear(H, 1).cuda()
        self.sigma = nn.Sigmoid()
        self.threshold = .99        
        # global variables
        self.updates = 0
        self.exit_ = torch.zeros(B, M, 1).long().cuda()
        self.run = torch.ones(B, M, 1).bool().cuda()
        # helper
        self.index_run = torch.arange(B*M).reshape(B, M, 1).cuda()
        self.align = torch.arange(M).cuda()
        # buffer
        self.buffer_unpack_h = torch.zeros(B, M, H).cuda()
        self.buffer_weighted_h = torch.zeros(B, M, H).cuda()
        self.buffer_acc_p = torch.zeros(B, M, 1).cuda()

    def init_act(self):
        # buffer
        self.buffer_unpack_h.zero_()
        self.buffer_weighted_h.zero_()
        self.buffer_acc_p.zero_()

        self.unpack_h = Variable(self.buffer_unpack_h)
        self.weighted_h = Variable(self.buffer_weighted_h)
        self.acc_p = Variable(self.buffer_acc_p)

        self.updates = 0
        self.exit_.zero_()
        self.run.fill_(True)

    def unpack(self, x, unpack_mask):
        """
        restore x to batch dimension
        with zero token in place of exit tokens
        [[C, C, C], => [[Z, C, C, Z, C]
         [C, E, E]]     [Z, Z, Z, Z, C]]
        C: Continue, E: Exit, Z: Zero 
        """
        B,M,H=self.unpack_h.size()
        # to do pad with pad token
        self.unpack_h.zero_()
        sum_ = unpack_mask.sum(1).view(-1)
        max_ = sum_.max()
        pack_mask = (-self.align[:max_] + sum_.unsqueeze(1))
        pack_mask = pack_mask.clamp(0, 1).bool()
        return (self.unpack_h.view(-1, H)
                .index_copy(0,
                            self.index_run[unpack_mask],
                            x[pack_mask]).view(B, M, H))

    def pack(self, x, unpack_mask):
        """
        left pack x to minimal dimension
        and pad with zero token
        [[E, C, C, E, C]  => [[C, C, C],
         [E, E, E, E, C]]     [C, Z, Z]]
        C: Continue, E: Exit, Z: Zero 
        """
        B,M,H=x.size()
        sum_ = unpack_mask.sum(1)
        max_ = sum_.max()
        pad = max_ - sum_.min()
        add_mask = (self.align[:pad] + 1 -
                    (sum_ - max_ + pad))
        add_mask = add_mask.clamp(0, 1).bool()
        pad_mask = torch.cat((unpack_mask.squeeze(), add_mask), 1)
        # to do pad with pad token
        x = F.pad(x, (0, 0, 0, pad))
        x = x[pad_mask].reshape(B, max_, H).contiguous()
        return x

    def forward(self, h):
        """ layer-wise act with left packing h """
        # unpack to full batch
        h = self.unpack(h, self.run)
        # exit probability
        p = self.sigma(self.p(h)) * self.run
        self.acc_p = self.acc_p + p
        # masks
        mask_continue = (self.acc_p < self.threshold) * self.run
        mask_exit = (~mask_continue) * self.run
        # current p or remainder
        p = p * mask_continue + (1 - (self.acc_p - p)) * mask_exit
        # ensure distribution on weighted_h
        self.weighted_h = (
            h * p +
            self.weighted_h * (1 - p)
        )
        # update containers
        self.run = self.run * mask_continue
        self.updates += 1
        self.exit_ = self.exit_ + self.updates * mask_exit
        # left pack h
        h = self.pack(h, self.run)
        return h
