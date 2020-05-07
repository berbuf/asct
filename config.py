#!/usr/bin/env python3
# command-line arguments with their default values
PARAMS_CONFIG_SMALL = {

    # env-specific
    'env_params': {
        '--device_num': {
            'type': int,
            'default': 2,
            'help': 'gpu num',
            'dest': 'device_num'
        },        
    },

    # data-specific
    'data_params': {
        '--data': {
            'type': str,
            'default': 'data/wiki/',
            'help': 'data location '
                    '(must contain train.txt, valid.txt and test.txt)',
            'dest': 'data_path'
        },
        '--vocab_size': {
            'type': int,
            'default': 30000,
            'help': 'size of vocabulary',
            'dest': 'vocab_size',
        },
    },

    # model-specific
    'model_params': {
        '--vanilla': {
            'type': bool,
            'default': False,
            'help': 'vanilla transformer',
            'dest': 'vanilla'
        },
        '--hid-sz': {
            'type': int,
            'default': 256,
            'help': 'hidden size (i.e. model size)',
            'dest': 'hidden_size'
        },
        '--inner-hid-sz': {
            'type': int,
            'default': 512,
            'help': 'inner hidden size of FF layer',
            'dest': 'inner_hidden_size'
        },
        '--nlayers': {
            'type': int,
            'default': 12,
            'help': 'number of layers',
            'dest': 'nb_layers'
        },
        '--block-sz': {
            'type': int,
            'default': 4096,
            'help': 'block size '
                    '(the length of sequence to process in parallel)',
            'dest': 'block_size'
        },
        '--nheads': {
            'type': int,
            'default': 8,
            'help': 'number of self-attention heads',
            'dest': 'nb_heads'
        },
        '--dropout': {
            'type': float,
            'default': 0.1,
            'help': 'dropout rate of ReLU and attention',
            'dest': 'dropout'
        },
        '--threshold_act': {
            'type': float,
            'default': .99,
            'help': 'threshold for act exit',
            'dest': 'threshold'
        },
    },

    # optimization-specific
    'optim_params': {
        '--lr': {
            'type': float,
            'default': 0.01,
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
            'default': 2,
            'help': 'batch size',
            'dest': 'batch_size'
        },
        '--last_iter': {
            'type': int,
            'default': 0,
            'help': '',
            'dest': 'last_iter'
        },
        '--niter': {
            'type': int,
            'default': 100,
            'help': 'number of iterations to train',
            'dest': 'nb_iter'
        },
        '--checkpoint': {
            'type': str,
            'default': 'IMDB',
            'help': 'path to save/load info model',
            'dest': 'checkpoint_path'
        },
        "--loss_act": {
            'type': float,
            'default': 0.0001,
            'help': '',
            'dest': 'loss_act'
        },
        "--loss_asa": {
            'type': float,
            'default': 0.4,
            'help': '',
            'dest': 'loss_asa'
        },
    },
}
