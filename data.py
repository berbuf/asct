import os
import torch
from os.path import join
from tokenizers import CharBPETokenizer
import pandas as pd
import re
import numpy as np

### ###
### IMDB
### ###
class ImdbCorpus:
    def __init__(self, train_data, train_label, val_data, val_label):
        self.train_data = train_data
        self.train_label = train_label
        self.val_data = val_data
        self.val_label = val_label

def get_data_vanilla(df, exp, token_id, cls_id, pad_id):
    entry_size = 512
    data, label = [], []
    for v,s in zip(df.review, df.sentiment):
        txt = re.findall(exp, v.lower())
        ids = cls_id + [ token_id[k] for k in txt ]
        if len(ids) > entry_size:
            ids = ids[:entry_size]
        ids += pad_id * (entry_size - len(ids))
        data += [ids]
        label += [ (0 if s == "negative" else 1) ]
    return data, label

def get_data_asct(df, exp, token_id, cls_id, pad_id, entry_size):
    def add_row(data, label, entry, entry_label, len_entry):
        flat_entry = [ l for e in entry for l in e ]
        flat_entry += pad_id * (entry_size - len_entry)
        data += [ flat_entry ]
        label += [ entry_label ]
        return data, label

    data, label, entry_num = [], [], 0
    len_entry, entry, entry_label  = 0, [], []
    for v,s in zip(df.review, df.sentiment):
        txt = re.findall(exp, v.lower())
        ids = [ token_id[k] for k in txt ]

        if len_entry + len(ids) + 1 > entry_size:
            data, label = add_row(data, label, entry, entry_label, len_entry)
            len_entry, entry, entry_label  = 0, [], []
            entry_num += 1

        entry += [ cls_id + ids ]
        entry_label += [ (len_entry, len_entry + len(ids) + 1,
                          0 if s == "negative" else 1) ]
        len_entry += len(ids) + 1

    data, label = add_row(data, label, entry, entry_label, len_entry)

    m = max([ len(l) for l in label ])
    label = [ l + [ (0, 0, -1) ] * (m - len(l)) for l in label ] # pad

    return data, label

def get_data_imdb(batch_size, device, data_path="./data/IMBD/",
                  entry_size=-1, train_share=0.5, vanilla=False, load=False):
    
    weights_embed, token_id, id_token = (
        create_matrix_embed_glove_imbd(dct=None, data_path=data_path, load=True))
    weights_embed = torch.LongTensor(weights_embed).to(device)

    if load and not vanilla:
        cp = torch.load(data_path + ".corpus.pt")
        return (weights_embed,
                (cp.train_data.to(device), cp.train_label.to(device)),
                (cp.val_data.to(device), cp.val_label.to(device)), token_id)

    if load:
        cp = torch.load(data_path + ".corpus_vanilla.pt")
        return (weights_embed,
                (cp.train_data.to(device), cp.train_label.to(device)),
                (cp.val_data.to(device), cp.val_label.to(device)), token_id)

    cls_id = [token_id["[CLS]"]]
    pad_id = [token_id["[PAD]"]]
    df = pd.read_csv(data_path + "IMBD.csv")
    exp = "[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\w]+"
    if vanilla:
        data, label = get_data_vanilla(df, exp, token_id, cls_id, pad_id)
    else:
        data, label = get_data_asct(df, exp, token_id, cls_id, pad_id, entry_size)

    data = torch.LongTensor(data).to(device)
    label = torch.LongTensor(label).to(device)

    train_share = int(len(data) * train_share)
    train_data, val_data = data[:train_share], data[:-train_share]
    train_label, val_label = label[:train_share], label[:-train_share]

    print ("train size:{}, val size:{}".format(len(train_data),
                                               len(val_data)))

    cp = ImdbCorpus(train_data, train_label, val_data, val_label)
    if vanilla:
        torch.save(cp, data_path + ".corpus.pt")
    else:
        torch.save(cp, data_path + ".corpus_vanilla.pt")

    return (weights_embed, (cp.train_data, cp.train_label),
            (cp.val_data, cp.val_label), token_id)

def create_matrix_embed_glove_imbd(dct, data_path="./data/IMBD/", load=True):
    if load:
        return (np.load(data_path + "glove.emb.npy", allow_pickle=True),
                np.load(data_path + "token_id.glove.emb.npy", allow_pickle=True).item(),
                np.load(data_path + "id_token.glove.emb.npy", allow_pickle=True).item())

    hidden_size = 300
    weights_matrix = []
    id_token = {}
    token_id = {}
    current_id = 0
    
    def add_line(k, current_id, glove_dct):
        id_token[current_id] = k
        token_id[k] = current_id
        if k not in glove_dct:
            weights = np.random.normal(scale=0.6, size=(hidden_size, ))
        else:
            weights = glove_dct[k]
        return id_token, token_id, current_id + 1, weights
        
    for k in ["[CLS]", "[PAD]", "[UNK]"]:
        id_token, token_id, current_id, weights = add_line(k, current_id, dct)
        weights_matrix += [weights]
        
    cls_id = token_id["[CLS]"]
    pad_id = token_id["[PAD]"]
    df = pd.read_csv(data_path + "IMBD.csv")
    exp = "[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\w]+"
    
    data, label = [], []
    len_entry, entry, entry_label  = 0, [], []
    for v,s in zip(df.review, df.sentiment):
        txt = re.findall(exp, v.lower())
        
        for k in txt:
            if k not in token_id:
                id_token, token_id, current_id, weights = add_line(k, current_id, dct)
                weights_matrix += [weights]

    print("Done.",len(weights_matrix)," size matrix")
    np.save(data_path + "glove.emb", weights_matrix)
    np.save(data_path + "token_id.glove.emb", token_id)
    np.save(data_path + "id_token.glove.emb", id_token)

def load_glove(path_name, file_name, load=True):
    if load:
        dct = np.load(path_name + "dct." + file_name + ".npy", allow_pickle=True)
        return dct.item()
    print("Loading Glove Model")
    f = open(path_name + file_name + ".txt",'r')
    dct = {}
    for i, line in enumerate(f):
        split_line = line.split()
        word = split_line[0]
        #print (word)
        embedding = np.array([float(val) for val in split_line[1:]])
        dct[word] = embedding
        if not i % 100000:
            print ("nb words:{}, current word:{}".format(i, word))
    print("Done.",len(dct)," words loaded")
    np.save(path_name + "dct." + file_name, dct)
    return dct

### ###
### BPE Tokenizer
### ###

def get_tokenizer(data_path, vocab_size):
    print ("Load tokenizer")
    return CharBPETokenizer(join(data_path, "tokenizer-uncased-vocab.json"),
                            join(data_path, "tokenizer-uncased-merges.txt"),
                            unk_token="[UNK]")

class Corpus:
    def __init__(self, data_path, vocab_size, tokenizer):
        assert os.path.exists(data_path)
        self.train = _tokenize(os.path.join(data_path, 'train.txt'),
                               tokenizer)
        self.train_mask = mask(self.train.clone(), tokenizer)
        self.valid = _tokenize(os.path.join(data_path, 'valid.txt'),
                               tokenizer)
        self.valid_mask = mask(self.valid.clone(), tokenizer)
        self.test = _tokenize(os.path.join(data_path, 'test.txt'),
                              tokenizer)
        self.test_mask = mask(self.test.clone(), tokenizer)

def _get_train_val_test_data(corpus, batch_size, device):
    return [
        (_batchify(corpus.train, batch_size).to(device),
         _batchify(corpus.train_mask, batch_size).to(device)),

        (_batchify(corpus.valid, batch_size).to(device),
         _batchify(corpus.valid_mask, batch_size).to(device)),

        (_batchify(corpus.test, batch_size).to(device),
         _batchify(corpus.test_mask, batch_size).to(device)),
    ]

def _build_corpus(data_path, vocab_size, env_params):
    tokenizer = get_tokenizer(data_path, vocab_size)
    # save the corpus to a file so that it's faster next time
    corpus_path = os.path.join(data_path, 'corpus.pt')
    if os.path.exists(corpus_path):
        print('Loading an existing corpus file from {}'.format(corpus_path))
        corpus = torch.load(corpus_path)
    else:
        print('Creating a corpus file at {}'.format(corpus_path))
        corpus = Corpus(data_path, vocab_size, tokenizer)
        torch.save(corpus, corpus_path)
    return corpus, tokenizer

def get_train_val_test_data(data_params, env_params, batch_size, device):
    corpus, tokenizer = _build_corpus(**data_params, env_params=env_params)
    train_data, val_data, test_data = _get_train_val_test_data(
        corpus=corpus, batch_size=batch_size, device=device)

    return train_data, val_data, test_data, tokenizer

def mask(text, mask_idx):
    # .15 IDX
    # .8 mask / .2 label
    B,M = text.shape
    perm = torch.randperm(M)
    label_idx = perm[:int(M*.15)]
    share = int(int(M*.15)*.8)
    text[:,label_idx[:share]] = mask_idx
    return text, label_idx

######
####
## WIKI CORPUS
def _batchify(data_tensor, batch_size):
    nb_batches = data_tensor.size(0) // batch_size
    # trim away some tokens to make whole batches
    data_tensor = data_tensor.narrow(0, 0, nb_batches * batch_size)
    data_tensor = data_tensor.view(batch_size, -1).contiguous()
    return data_tensor

def _tokenize(text_path, tokenizer):
    """Tokenizes a text file."""
    with open(text_path, 'r', encoding="utf8") as f:
        ids = tokenizer.encode("".join(f.readlines())).ids
        #ids = tokenizer.encode(f.read()).ids
    print ("{:<30}, Nb tokens: {} ".format(text_path, len(ids)))
    return torch.LongTensor(ids)

class WikiCorpus:
    def __init__(self, data_path, batch_size, device, tokenizer):
        self.train = _tokenize(data_path, tokenizer)        

def get_data_block(data_path, batch_size, device, tokenizer, file):
    data_path = "{}{}".format(data_path, file)
    corpus_path = "{}_{}_{}".format(data_path, str(batch_size), "corpus.pt")
    if os.path.exists(corpus_path):
        print('Loading an existing corpus file from {}'.format(corpus_path))
        corpus = torch.load(corpus_path)
    else:
        print('Creating a corpus file at {}'.format(corpus_path))
        corpus = WikiCorpus(data_path, batch_size, device, tokenizer)
        torch.save(corpus, corpus_path)
    return _batchify(corpus.train, batch_size).to(device)
