#!/usr/bin/env python3
# command-line arguments with their default values

PARAMS_CONFIG_TASKS = {
    "cola": { 'data_path': "glue_data/CoLA/", "num_epoch": 3, "cls_layer": 2, "batch_size": 64 // 4,
              "optim_params": {"lr": 2e-3, "optim": "sgd", "momentum": 0.9, "grad_clip": 0, "lr_warmup": 0}
    },
    "mnli": { 'data_path': "glue_data/MNLI/",
              "optim_params": {"lr": 2e-5, "optim": "sgd", "momentum": 0.9, "grad_clip": 0, "lr_warmup": 0}
    },
    "sst-2": { 'data_path': "glue_data/SST-2/",
               "optim_params": {"lr": 2e-5, "optim": "sgd", "momentum": 0.9, "grad_clip": 0, "lr_warmup": 0}
    },
    "sts-b": { 'data_path': "glue_data/STS-B/",
              "optim_params": {"lr": 2e-5, "optim": "sgd", "momentum": 0.9, "grad_clip": 0, "lr_warmup": 0}
    },
    "qnli": { 'data_path': "glue_data/QNLI/",
              "optim_params": {"lr": 2e-5, "optim": "sgd", "momentum": 0.9, "grad_clip": 0, "lr_warmup": 0}
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
