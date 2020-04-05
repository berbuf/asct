from tokenizers import CharBPETokenizer
import os

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


#python -m gensim.scripts.segment_wiki -i -f enwiki-latest-pages-articles.xml.bz2 -o enwiki-latest.json.gz
#split -C 100000000 -d wikitext-103/train.txt chunk

#dump_text_wiki(min_size=256, file_path="./clean_wiki.txt")
#append_book_corpus()
create_tokenizer(data_path="./data/wiki/", vocab_size=30000)
