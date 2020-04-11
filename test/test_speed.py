import time
import torch
from models import TransformerLayer

from vanilla_model import TransformerSeqLayer as SeqLayer


device=3
torch.cuda.set_device(device)

iter=64
N=12
V=1000

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
    print (" std ", end - start)


def asa_layer(B,M,H,K,V,L):
    D = H//K
    emb = torch.nn.Embedding(V, H).cuda()
    X = torch.arange(B*M).reshape(B,M).cuda() % V

    layer = TransformerLayer(B, M, H, K,
                             **{"dropout":.1, "inner_hidden_size":H*2, "span":L}).cuda()

    pad_idx = torch.tensor([0]).cuda()

    start = time.time()
    for _ in range(iter):
        h = emb(X)

        pad_h = emb(pad_idx).unsqueeze(1) # 1, 1, H
        pad_h = pad_h.repeat(B,L,1) # B x L x H
        pad_h = layer.attn.head_reshape(pad_h, D) # BK x L x D
        layer.attn.attn.select.set_pad_h(pad_h)

        for _ in range(N):
            h = layer(h)

        h = h.detach()

    end = time.time()
    print (" asa ", end - start)


B,M,H,K,L=1,4096,128,2,32

for M in [4096, 5120, 6144, 7168, 8192]:
    print (M)
    asa_layer(B=1,M=M,H=128,K=8,V=1000,L=256)

"""
for H in [256, 512, 1024]:
    
    for K in [2, 4, 8]:
        
        for L in [32, 64, 128, 256]:
            
            print ("M:{}, H:{}, K:{}, L:{}".format(M,H,K,L))
            #std_layer(B,M,H,K,V,L)
            asa_layer(B,M,H,K,V,L)
"""

"""
for M in [512, 1024, 2048, 4096]:

    for H in [128, 256, 512, 1024]:

        for K in [2, 4, 8, 16]:

            for L in [32, 64, 128, 256]:

                print ("M:{}, H:{}, K:{}, L:{}".format(M,H,K,L))
                std_layer(B,M,H,K,V,L)
                asa_layer(B,M,H,K,V,L)
"""

            #break
        #break
    #break
