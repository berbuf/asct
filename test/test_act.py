import time
import torch

from vanilla_model import TransformerSeqLayer as SeqLayer
from models import FeedForwardLayer
from act import AdaptiveComputationTime

device=3
torch.cuda.set_device(device)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

iter=64
N=12

def std_layer(B,M,H,K,V,L):
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
    print ("no_act", end - start)
    
def act_layer(B,M,H,K,V,L):

    emb = torch.nn.Embedding(V, H).cuda()
    X = torch.arange(B*M).reshape(B,M).cuda() % V

    act = AdaptiveComputationTime(B,M,H,0.99).cuda()
    layer = SeqLayer(H, **{"inner_hidden_size": H*2, "dropout": .1, "nb_heads":K }).cuda()
    key_pe = torch.nn.Parameter(torch.randn(1, H//K, M)).cuda()

    pad_idx = torch.tensor([0]).cuda()

    start = time.time()
    for _ in range(iter):
        h = emb(X)

        pad_h = emb(pad_idx)[0]
        act.init_act(pad_h)

        for i in range(N):    
            print (h.shape)
            B,M,H=h.size()
            if not M:
                break
            h = layer(h, key_pe)

            d=1/(i+1)
            h = act(h, d*.8)
        print ()
        h = h.detach()

    end = time.time()
    print ("act", end - start)


B,M,H,K,L=1,512,512,8,32
V=1000
D=H//K

act_layer(B,M,H,K,V,L=-1)

"""
for M in [512, 1024, 2048, 4096]:

    for H in [128, 256, 512, 1024]:
        print ("M:{}, H:{}, K:{}".format(M,H,K))
        std_layer(B,M,H,K,V,L=-1)
        act_layer(B,M,H,K,V,L=-1)
            
"""

