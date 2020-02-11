import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveComputationTime(nn.Module):

    def __init__(self, block_size, hidden_size,
                 dup_batch_size, **kargs):
        nn.Module.__init__(self)
        self.p = nn.Linear(hidden_size, 1).cuda()
        self.sigma = nn.Sigmoid()
        self.threshold = .99

    def init_batch(self, h):
        B,M,H=h.size()
        # final distribution on weights
        self.weighted_h = torch.zeros(B, M, H).cuda()
        self.acc_p = torch.zeros(B, M, 1).cuda()
        # N and remainder
        self.updates = 0
        self.remainders = torch.zeros(B, M, 1).cuda()
        # global mask
        self.run = torch.ones(B, M, 1).bool().cuda()
        # helper for masks
        self.index_run = torch.arange(B*M).reshape(B, M, 1).cuda()
        self.unpack_weights = torch.zeros(B, M, H).cuda()
        self.align = torch.arange(M).cuda()

    def unpack(self, x, unpack_mask):
        """
        restore x to batch dimension
        with zero token in place of exit tokens
        [[C, C, C], => [[Z, C, C, Z, C]
         [C, E, E]]     [Z, Z, Z, Z, C]]
        C: Continue, E: Exit, Z: Zero 
        """
        _,_,H=x.size()
        # to do pad with pad token
        self.unpack_weights.fill_(0)
        sum_ = unpack_mask.sum(1).view(-1)
        max_ = sum_.max()
        pack_mask = (-self.align[:max_] + (sum_).unsqueeze(1))
        pack_mask = pack_mask.clamp(0, 1).bool()
        (self.unpack_weights.view(-1, H)
         .index_copy_(0,
                      self.index_run[unpack_mask],
                      x[pack_mask]))
        return self.unpack_weights

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
        self.acc_p += p
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
        self.run *= mask_continue
        self.remainders += p * mask_exit
        self.updates += 1
        # left pack h
        h = self.pack(h, self.run)
        return h

    def loss(self):
        """ minimize number of updates and remaining probability """
        return self.updates, self.remainders

def test():
    #B,M,H=64,512,1024
    torch.cuda.set_device(3)
    B,M,H=2,10,3
    x = torch.FloatTensor(B, M, H).uniform_(0, 10).float().cuda()
    act = AdaptiveComputationTime(M, H, B)
    act.init_batch(x)
    while M > 0:
        x = act(x)
        x.uniform_(0, 10)
        _,M,_=x.size()
        print (M)

test()
