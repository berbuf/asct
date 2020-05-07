from tokenizers import CharBPETokenizer
import os
import torch

def create_tokenizer(data_path, vocab_size):

    tokenizer = CharBPETokenizer()
    tokenizer.train([ os.path.join(data_path, file) for file in
                      [ f for f in os.listdir(data_path)
                        if f.find("uncased_chunk") != -1 ][:20] ],
                    vocab_size=vocab_size,
                    min_frequency=2,
                    show_progress=True,
                    special_tokens=["[CLS]", "[PAD]", "[MASK]", "[UNK]", "[SEP]"])

    print ("[CLS]: {}, [PAD]: {}, [MASK]: {}, [UNK]: {}, [SEP]: {}".format(
        str(tokenizer.token_to_id("[CLS]")),
        str(tokenizer.token_to_id("[PAD]")),
        str(tokenizer.token_to_id("[MASK]")),
        str(tokenizer.token_to_id("[UNK]")),
        str(tokenizer.token_to_id("[SEP]"))))

    tokenizer.save(data_path, "tokenizer")

from gensim import utils
import json
import numpy as np

def dump_text_wiki(min_size, file_path):
    with utils.open(file_path, 'a') as fw:
        with utils.open('./data/wiki/enwiki-latest.json.gz', 'rb') as fr:
            c = 0
            len_ = []
            for line in fr:
                article = json.loads(line)
                s = []
                for text in article['section_texts']:
                    s += [text]
                text_article = "\n".join(s)
                l = len(text_article.split(" "))
                if l < 256:
                    continue
                fw.write(text_article + "[END]")
                len_ += [l]
                c += 1
                if not c % 100000:
                    print ("wrote", c, "articles.", "Mean", np.mean(len_))
                    len_ = []

import os
def append_book_corpus():
    path = "./data/wiki/book_corpus/"
    with utils.open("./data/wiki/clean_wiki.txt", 'a') as fw:
        for file in os.listdir(path):
            if file.find(".epub") != -1:
                continue
            with utils.open(path + file, 'r') as fr:
                book_text = fr.read()
                fw.write(book_text + "[END]")

def create_tokenizer_imbd(data_path, file_name, vocab_size):
    #df = pd.read_csv(os.path.join(data_path, file_name))
    tokenizer = CharBPETokenizer()
    tokenizer.train(os.path.join(data_path, file_name),
                    vocab_size=vocab_size,
                    min_frequency=2,
                    show_progress=True,
                    special_tokens=["[CLS]", "[PAD]", "[MASK]", "[UNK]", "[SEP]"])

    print ("[CLS]: {}, [PAD]: {}, [MASK]: {}, [UNK]: {}, [SEP]: {}".format(
        str(tokenizer.token_to_id("[CLS]")),
        str(tokenizer.token_to_id("[PAD]")),
        str(tokenizer.token_to_id("[MASK]")),
        str(tokenizer.token_to_id("[UNK]")),
        str(tokenizer.token_to_id("[SEP]"))))

    tokenizer.save(data_path, "tokenizer")

import pandas as pd

def get_data_imdb(batch_size, device, data_path="./data/IMBD/",
                  entry_size=8192, train_share=0.8):
    glove_dct = load_glove("./data/glove/", "glove.42B.300d", load=True)
    weights_matrix, token_id, id_token = (
        create_matrix_embed_glove_imbd(glove_dct, data_path, load=True))

    cls_id = [token_id("[CLS]")]
    pad_id = [token_id("[CLS]")]

    df = pd.read_csv(data_path + "IMBD.csv")

    def add_row(data, label, entry, entry_label):
        flat_entry = [ l for e in entry for l in e ]
        flat_entry += pad_id * (entry_size - len_entry)
        data += [ flat_entry ]
        label += [[ entry_label ]]
        return data, label

    exp = "[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\w]+"

    data, label = [], []
    len_entry, entry, entry_label  = 0, [], []
    for v,s in zip(df.review, df.sentiment):
        txt = re.findall(exp, v.lower())
        ids = [ token_id[k] for k in txt ]

        if len_entry + len(ids) + 1 > entry_size:
            data, label = add_row(data, label, entry, entry_label)
            len_entry, entry, entry_label  = 0, [], []

        entry += [ cls_id + ids ]
        l = 0 if s == "negative" else 1
        entry_label += [ (len_entry, l) ]
        len_entry += len(ids) + 1

    data, label = add_row(data, label, entry, entry_label)

    train_share = int(len(data) * train_share)
    train_data, val_data = data[:train_share], data[:-train_share]
    train_label, val_label = label[:train_share], label[:-train_share]
    
    print ("train size:{}, val size:{}".format(len(train_data),
                                               len(val_data)))
    f = lambda x: torch.LongTensor(x).to(device)
    return ((f(train_data), train_label),
            (f(val_data), val_label),
    tok)

import re
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

#model = load_glove("./data/glove/", "glove.42B.300d", load=False)

#read_imbd(dct)

#create_tokenizer_imbd(data_path="./data/IMBD/", file_name="IMBD.csv", vocab_size=30000)

#python -m gensim.scripts.segment_wiki -i -f enwiki-latest-pages-articles.xml.bz2 -o enwiki-latest.json.gz
#split -C 100000000 -d wikitext-103/train.txt chunk

#dump_text_wiki(min_size=256, file_path="./clean_wiki.txt")
#append_book_corpus()
#create_tokenizer(data_path="./data/wiki/", vocab_size=30000)
