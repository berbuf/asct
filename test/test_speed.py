import time
import torch
from models import TransformerLayer

from vanilla_model import TransformerSeqLayer as SeqLayer


device=1
torch.cuda.set_device(device)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

iter=64
N=12
V=1000

def std_layer(B,M,H,K,V):
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


def asa_layer(B,M,H,K,V):
    D = H//K
    emb = torch.nn.Embedding(V, H).cuda()
    X = torch.arange(B*M).reshape(B,M).cuda() % V

    layer = TransformerLayer(B, M, H, K,
                             **{"dropout":.1, "inner_hidden_size":H*2}).cuda()

    pad_idx = torch.tensor([0]).cuda()
    start = time.time()
    for _ in range(iter):
        h = emb(X)

        pad_h = emb(pad_idx).unsqueeze(1) # 1, 1, H
        pad_h = pad_h.repeat(B,1,1) # B x 1 x H
        pad_h = layer.attn.head_reshape(pad_h, D) # BK x 1 x D
        layer.attn.attn.select.set_pad_h(pad_h)

        for _ in range(N):
            h = layer(h)

        h = h.detach()

    end = time.time()
    print (" asa ", end - start)


B,M,H,K=1,4096,1024,8

print ("M:{}, H:{}, K:{}".format(M,H,K))
asa_layer(B=B,M=M,H=H,K=K,V=V)
std_layer(B=B,M=M,H=H,K=K,V=V)

