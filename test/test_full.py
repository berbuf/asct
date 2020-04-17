import time
import torch

from vanilla_model import TransformerSeqLayer as SeqLayer
from models import TransformerLayer
from act import AdaptiveComputationTime

device=0
torch.cuda.set_device(device)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

iter=64
N=12
V=1000

def std_layer(B,M,H,K):
    emb = torch.nn.Embedding(V, H).cuda()
    X = torch.arange(B*M).reshape(B,M).cuda() % V

    layer = SeqLayer(H, **{"inner_hidden_size": H*2, "dropout": .1, "nb_heads":K }).cuda()
    key_pe = torch.nn.Parameter(torch.randn(1, H//K, M)).cuda()

    start = time.time()
    for _ in range(iter):
        
        h = emb(X)
        for _ in range(N):    
            h = layer(h, key_pe)
        h = h.detach()

    end = time.time()
    print ("std", end - start)

def comp_layer(B,M,H,K):
    D=H//K

    emb = torch.nn.Embedding(V,H).cuda()
    X = torch.arange(B*M).reshape(B,M).cuda() % V

    act = AdaptiveComputationTime(B,M,H,0.99).cuda()
    layer = TransformerLayer(B,M,H,K,
                             **{"dropout":.1, "inner_hidden_size":H*2}).cuda()

    pad_idx = torch.tensor([0]).cuda()

    start = time.time()
    for _ in range(iter):
        h = emb(X)

        pad_h = emb(pad_idx).unsqueeze(1) # 1, 1, H

        # set act
        act.init_act(pad_h.squeeze())

        # set asa
        pad_h = pad_h.repeat(B,1,1) # B x 1 x H
        pad_h = layer.attn.head_reshape(pad_h, D) # BK x 1 x D
        layer.attn.attn.select.set_pad_h(pad_h)

        for i in range(N):    
            print (h.shape)
            B,M,H=h.size()
            if not M:
                break
            h = layer(h)
            d=1/(i+1)
            h = act(h, d*2.)

        print ()
        h = h.detach()

    end = time.time()
    print ("comp", end - start)


B,M,H,K=1,512,128,16

print ("M:{}, H:{}, K:{}".format(M,H,K))
#std_layer(B,M,H,K)
comp_layer(B,M,H,K)
#print ("M:{}, H:{}, K:{}".format(M,H,K))
#std_layer(B,M,H,K)
#comp_layer(B,M,H,K)

"""
for M in [512, 1024, 2048, 4096]:

    for H in [128, 256, 512, 1024]:

        print ("M:{}, H:{}, K:{}".format(M,H,K))
        if M == 4096 and H == 1024:
            print ("std no")
            comp_layer(B,M=M,H=H,K=K)
        else:
            std_layer(B,M=M,H=H,K=K)
            comp_layer(B,M=M,H=H,K=K)
"""
