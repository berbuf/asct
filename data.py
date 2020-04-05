import os
import torch
from os.path import join
from tokenizers import CharBPETokenizer

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

    if env_params['distributed']:
        # split the data into equal parts
        assert batch_size % env_params['world_size'] == 0
        device_batch_size = batch_size // env_params['world_size']
        slice_data = slice(
            device_batch_size * env_params['rank'],
            device_batch_size * (env_params['rank'] + 1))
        train_data = train_data[slice_data]
        val_data = val_data[slice_data]
        test_data = test_data[slice_data]

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
