#!/usr/bin/env python3
# command-line arguments with their default values

PARAMS_CONFIG_TASKS = {
    "qnli": { 'data_path': "glue_data/QNLI/", "num_epoch": 3, "batch_size": 64 // 2, "block_size": 256,
              "optim_params": {"lr": 2e-5, "optim": "sgd", "momentum": 0.9, "grad_clip": 0, "lr_warmup": 0}
    },
    "sts-b": { 'data_path': "glue_data/STS-B/", "num_epoch": 3, "batch_size": 64 // 2, "block_size": 256,
              "optim_params": {"lr": 2e-5, "optim": "sgd", "momentum": 0.9, "grad_clip": 0, "lr_warmup": 0}
    },
    "cola": { 'data_path': "glue_data/CoLA/", "num_epoch": 3, "batch_size": 64 // 2, "block_size": 256,
              "optim_params": {"lr": 2e-5, "optim": "sgd", "momentum": 0.9, "grad_clip": 0, "lr_warmup": 0}
    },
    "sst-2": { 'data_path': "glue_data/SST-2/", "num_epoch": 3, "batch_size": 64 // 2, "block_size": 256,
               "optim_params": {"lr": 2e-3, "optim": "sgd", "momentum": 0.9, "grad_clip": 0, "lr_warmup": 0}
    },
    "mnli": { 'data_path': "glue_data/MNLI/", "num_epoch": 3, "batch_size": 64 // 4, "block_size": 512,
              "optim_params": {"lr": 2e-3, "optim": "sgd", "momentum": 0.9, "grad_clip": 0, "lr_warmup": 0}
    },
    #"mnli-mm": { 'data_path': "glue_data/CoLA/", "max_seq_length": 512,
    #},
    #"mrpc": { 'data_path': "glue_data/MRPC/", "max_seq_length": 512,
    #},
    #"qqp": { 'data_path': "glue_data/CoLA/", "max_seq_length": 512,
    #},
    #"rte": { 'data_path': "glue_data/CoLA/", "max_seq_length": 512,
    #},
    #"wnli": { 'data_path': "glue_data/CoLA/", "max_seq_length": 512, "lr": 2e-5,
    #},
}
