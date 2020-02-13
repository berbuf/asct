#!/usr/bin/env python3

# command-line arguments with their default values

PARAMS_CONFIG = {
    # env-specific
    'env_params': {
        '--distributed': {
            'action': 'store_true',
            'default': False,
            'help': 'enable distributed training.'
                    '(otherwise will use all available GPUs with dataparallel)',
            'dest': 'distributed'
        },
        '--local_rank': {
            'type': int,
            'default': 0,
            'help': 'used in distributed training',
            'dest': 'local_rank'
        },
        '--device_num': {
            'type': int,
            'default': 0,
            'help': 'gpu num',
            'dest': 'device_num'
        },
    },
    # data-specific
    'data_params': {
        '--data': {
            'type': str,
            'default': 'data/wikitext-2',
            'help': 'data location '
                    '(must contain train.txt, valid.txt and test.txt)',
            'dest': 'data_path'
        },
        '--tokenize': {
            'action': 'store_true',
            'default': True,
            'help': 'retokenize corpus with bpe and save tokenizer',
            'dest': 'tokenize',
        },
        '--vocab_size': {
            'type': int,
            'default': 20000,
            'help': 'size of vocabulary',
            'dest': 'vocab_size',
        },
    },
    # model-specific
    'model_params': {
        '--dup-batch-sz': { # duplicate, nedded in SelectAttention
            'type': int,
            'default': 64,
            'help': 'duplicate batch size',
            'dest': 'dup_batch_size'
        },
        '--hid-sz': {
            'type': int,
            'default': 1024,
            'help': 'hidden size (i.e. model size)',
            'dest': 'hidden_size'
        },
        '--inner-hid-sz': {
            'type': int,
            'default': 1024,
            'help': 'inner hidden size of FF layer',
            'dest': 'inner_hidden_size'
        },
        '--nlayers': {
            'type': int,
            'default': 3,
            'help': 'number of layers',
            'dest': 'nb_layers'
        },
        '--block-sz': {
            'type': int,
            'default': 256,
            'help': 'block size '
                    '(the length of sequence to process in parallel)',
            'dest': 'block_size'
        },
        '--nheads': {
            'type': int,
            'default': 2,
            'help': 'number of self-attention heads',
            'dest': 'nb_heads'
        },
        '--rev-net': {
            'action': 'store_true',
            'default': True,
            'help': 'activate reversible network '
                    'to avoid storing gradients',
            'dest': 'rev_net'
        },
        '--dropout': {
            'type': float,
            'default': 0.2,
            'help': 'dropout rate of ReLU and attention',
            'dest': 'dropout'
        },
        '--soft': {
            'type': float,
            'default': 2.,
            'help': 'soft parameter '
                    'in select attention function',
            'dest': 'soft'
        },
        '--context_loss_scale': {
            'type': float,
            'default': .1,
            'help': 'scale (std dev) parameter '
                    'for contextual loss',
            'dest': 'context_loss_scale'
        },
    },
    # optimization-specific
    'optim_params': {
        '--lr': {
            'type': float,
            'default': 0.03,
            'help': 'learning rate',
            'dest': 'lr'
        },
        '--momentum': {
            'type': float,
            'default': 0.9,
            'help': 'SGD momentum',
            'dest': 'momentum'
        },
        '--optim': {
            'type': str,
            'default': 'sgd',
            'help': 'optimization method: sgd | adagrad',
            'dest': 'optim'
        },
        '--lr-warmup': {
            'type': int,
            'default': 0,
            'help': 'linearly increase LR from 0 '
                    'during first lr_warmup updates',
            'dest': 'lr_warmup'
        },
        '--grad-clip': {
            'type': float,
            'default': 0,
            'help': '[only works with adagrad!] '
                    'clip gradient of each module parameters by a given '
                    'value',
            'dest': 'grad_clip'
        },
    },
    # trainer-specific
    'trainer_params': {
        '--batch-sz': {
            'type': int,
            'default': 64,
            'help': 'batch size',
            'dest': 'batch_size'
        },
        '--batch-split': {
            'type': int,
            'default': 1,
            'help': 'split a batch into smaller parts to fit in GPU memory',
            'dest': 'batch_split'
        },
        '--nbatches': {
            'type': int,
            'default': 1000,
            'help': 'number of batches in each iteration',
            'dest': 'nb_batches_per_iter'
        },
        '--niter': {
            'type': int,
            'default': 1000,
            'help': 'number of iterations to train',
            'dest': 'nb_iter'
        },
        '--checkpoint': {
            'type': str,
            'default': '',
            'help': 'path to save/load model',
            'dest': 'checkpoint_path'
        },
        '--full-eval-mode': {
            'action': 'store_true',
            'default': True,
            'help': 'do evaluation on the whole validation and the test data',
            'dest': 'full_eval_mode'
        },
    },
}
