import torch.nn.functional as F
import torch
import torch.nn as nn
from scipy.signal import gaussian

def _skew(X):
    """ skew each row acumulatively with 0 """
    B,M,V=X.size()
    X = F.pad(X, (0, M + 1))
    X = X.view(B, -1)
    X = X[:, :-M]
    X = X.view(B, M, M + V)
    return X

class ContextualLoss(object):

    def __init__(self, vocab_size, batch_size,
                 block_size, context_loss_scale, **kargs):
        self.max_scale = 0
        self.scale = context_loss_scale
        self.windows = torch.tensor([]).float()
        self.p_voc = (torch.zeros(batch_size,
                                  block_size, vocab_size)
                      .cuda())

    def gaussian_windows(self, exit_token):
        B,M,_=exit_token.size()
        batch_max_scale = exit_token.max().int().item()
        if batch_max_scale > self.max_scale:
            app_windows = (torch.tensor(
                [ gaussian(M * 2 + 1, std=self.scale * std)
                  for std in range(self.max_scale, batch_max_scale + 1) ])
                           .float().cuda())
            self.windows = (app_windows if not len(self.windows)
                            else torch.cat((self.windows, app_windows)))
            self.max_scale = batch_max_scale
        return self.windows[exit_token.squeeze().long() - 1] # -1: (start at 0)

    def loss(self, log_p, index_label, exit_token):
        B,M,V=log_p.size()
        p_label = self.gaussian_windows(exit_token)
        p_label = _skew(p_label)
        p_label = p_label[:,:, M : -M - 1]
        # build p_voc by summing p_label
        # faster way ?
        self.p_voc.fill_(0)
        for b in range(B):
            self.p_voc[b].index_add_(1, index_label[b], p_label[b])
        # ensure probability
        self.p_voc /= self.p_voc.sum(2).unsqueeze(2) # B X M X V
        # cross entropy
        # -(\Sum { p^label_i * log(p^out_i) })
        loss = -self.p_voc.mul(log_p).sum(2) # B X M
        return loss.mean()
