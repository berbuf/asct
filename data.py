import os
import torch
from os.path import join
from tokenizers import CharBPETokenizer

from utils import check_ram

def get_tokenizer(data_path, vocab_size):
    tok_voc = os.path.join(data_path, 'tokenizer-vocab.json')
    tok_merg = os.path.join(data_path, 'tokenizer-merges.json')
    if os.path.exists(tok_voc) and os.path.exist(tok_merg):
        print ("Load tokenizer")
        return CharBPETokenizer(join(data_path, "tokenizer-vocab.json"),
                            join(data_path, "tokenizer-merges.txt"))
    print ("Create tokenizer")
    print ("Data path", join(data_path, 'train.txt'))
    tokenizer = CharBPETokenizer()
    tokenizer.train([
        join(data_path, 'train.txt'),
        join(data_path, 'valid.txt'),
        join(data_path, 'test.txt')
    ],
                    vocab_size=vocab_size,
                    min_frequency=2,
                    show_progress=True,
                    special_tokens=["[CLS]", "[PAD]", "[MASK]", "[UNK]"],
    )
    print ("[CLS]: {}, [PAD]: {}, [MASK]: {}, [UNK]: {}".format(
        str(tokenizer.token_to_id("[CLS]")),
        str(tokenizer.token_to_id("[PAD]")),
        str(tokenizer.token_to_id("[MASK]")),
        str(tokenizer.token_to_id("[UNK]"))))
    tokenizer.save(data_path, "tokenizer")
    return tokenizer

def _tokenize(text_path, tokenizer):
    """Tokenizes a text file."""
    with open(text_path, 'r', encoding="utf8") as f:
        ids = tokenizer.encode(". ".join(f.readlines())).ids
    print ("{:<30}, Nb tokens: {} ".format(text_path, len(ids)))
    return torch.LongTensor(ids)

def mask(text, tokenizer):
    size = text.size(0)
    perm = torch.randperm(size)
    perm = perm[:int(size*.15)]
    text[perm] = tokenizer.token_to_id("[MASK]")
    return text

class Corpus:
    def __init__(self, data_path, vocab_size):
        assert os.path.exists(data_path)
        tokenizer = get_tokenizer(data_path, vocab_size)
        self.train = _tokenize(os.path.join(data_path, 'train.txt'),
                               tokenizer)
        self.train_mask = mask(self.train.clone(), tokenizer)
        self.valid = _tokenize(os.path.join(data_path, 'valid.txt'),
                               tokenizer)
        self.valid_mask = mask(self.valid.clone(), tokenizer)
        self.test = _tokenize(os.path.join(data_path, 'test.txt'),
                              tokenizer)
        self.test_mask = mask(self.test.clone(), tokenizer)

def _batchify(data_tensor, batch_size):
    nb_batches = data_tensor.size(0) // batch_size
    # trim away some tokens to make whole batches
    data_tensor = data_tensor.narrow(0, 0, nb_batches * batch_size)
    data_tensor = data_tensor.view(batch_size, -1).contiguous()
    return data_tensor

def _build_corpus(data_path, vocab_size, env_params):
    # save the corpus to a file so that it's faster next time
    corpus_path = os.path.join(data_path, 'corpus.pt')
    if os.path.exists(corpus_path):
        print('Loading an existing corpus file from {}'.format(corpus_path))
        corpus = torch.load(corpus_path)
    else:
        print('Creating a corpus file at {}'.format(corpus_path))
        if env_params['distributed']:
            # only one process need to create a corpus file
            if env_params['rank'] == 0:
                corpus = Corpus(data_path, vocab_size)
                torch.save(corpus, corpus_path)
                # sync with other processes
                torch.distributed.broadcast(torch.zeros(1).cuda(), src=0)
            else:
                print('Waiting rank0 to create a corpus file.')
                # sync with rank0
                torch.distributed.broadcast(torch.zeros(1).cuda(), src=0)
                corpus = torch.load(corpus_path)
        else:
            corpus = Corpus(data_path, vocab_size)
            torch.save(corpus, corpus_path)
    return corpus

def _get_train_val_test_data(corpus, batch_size, device):
    return [
        (_batchify(corpus.train, batch_size).to(device),
         _batchify(corpus.train_mask, batch_size).to(device)),

        (_batchify(corpus.valid, batch_size).to(device),
         _batchify(corpus.valid_mask, batch_size).to(device)),

        (_batchify(corpus.test, batch_size).to(device),
         _batchify(corpus.test_mask, batch_size).to(device)),
    ]

def get_train_val_test_data(data_params, env_params, batch_size, device):
    corpus = _build_corpus(**data_params, env_params=env_params)
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

    return train_data, val_data, test_data
