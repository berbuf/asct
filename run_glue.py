#!/usr/bin/env python3

import math
import time
from os.path import join, exists

import torch

from config import PARAMS_CONFIG_SMALL as CONFIG
from task_config import PARAMS_CONFIG_TASKS as TASK_CONFIG

from data import get_train_val_test_data
from models import GenDisc, AsctSequenceClassification

from trainer import train_iteration, full_eval
from utils import (get_params, set_up_env,
                   get_optimizer_and_scheduler,
                   load_checkpoint, save_checkpoint, Logger)

from glue.processors.glue import glue_processors as processors
from glue.processors.glue import glue_output_modes as output_modes
from glue.metrics import glue_compute_metrics as compute_metrics
from glue.processors.glue import glue_convert_examples_to_features as convert_examples_to_features

from tokenizers import CharBPETokenizer

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import numpy as np

def train_glue(train_data, val_data, main_params, trainer_params, env_params, task_config):
    model = main_params["model"]
    optimizer = main_params["optimizer"]
    scheduler = main_params["scheduler"]

    ## Train
    model.train()
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=trainer_params["batch_size"])
    epochs, acc_loss, nb_batches  = 0, 0., 0.
    for e in range(epochs, task_config["num_epoch"]):
        print ("epoch", e)

        for step, batch in enumerate(train_dataloader):

            # ensure same batch_size
            if trainer_params["batch_size"] != len(batch[0]):
                continue

            # clear gradient
            optimizer.zero_grad()
            
            batch = tuple(t.to(env_params["device"]) for t in batch)

            loss = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[3])

            acc_loss += loss.item()
            nb_batches += 1

            print(loss.item())
            if torch.isnan(loss):
                return

            # step +1
            loss.backward()
            print ("GRAD")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print ()
                    print (name)
                    print (param.data.shape)

                    if param.grad is None:
                        print ("Val: None")
                    else:
                        if len(param.grad.shape) > 1:
                            print (param.grad[:10, :10])
                        else:
                            print (param.grad[:10])
            return

            #scheduler.step()
            optimizer.step()

        print ("epoch loss:", loss / nb_batches)
            

def load_glue(tokenizer, task, task_config):
    data_path = task_config["data_path"]

    processor = processors[task]()
    output_mode = output_modes[task]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    file_name_train = "{}_{}_{}".format("train", task_config["block_size"], task)
    file_name_dev = "{}_{}_{}".format("dev", task_config["block_size"], task)

    train_path = join(data_path, file_name_train)
    dev_path = join(data_path, file_name_dev)

    if exists(train_path):
        train = torch.load(train_path)
        dev = torch.load(dev_path)
    else:
        train = processor.get_train_examples(data_path)
        dev = processor.get_dev_examples(data_path)

        train = convert_examples_to_features(
            train, tokenizer, label_list=label_list,
            max_length=task_config["block_size"],
            output_mode=output_mode)
        dev = convert_examples_to_features(
            dev, tokenizer, label_list=label_list,
            max_length=task_config["block_size"],
            output_mode=output_mode)

        torch.save(train, train_path)
        torch.save(dev, dev_path)

    # Convert to Tensors and build dataset
    train_input_ids = torch.tensor([f.input_ids for f in train], dtype=torch.long)
    dev_input_ids = torch.tensor([f.input_ids for f in dev], dtype=torch.long)

    train_attention_mask = torch.tensor([f.attention_mask for f in train], dtype=torch.long)
    dev_attention_mask = torch.tensor([f.attention_mask for f in dev], dtype=torch.long)

    train_token_type_ids = torch.tensor([f.token_type_ids for f in train], dtype=torch.long)
    dev_token_type_ids = torch.tensor([f.token_type_ids for f in dev], dtype=torch.long)

    if output_mode == "classification":
        train_labels = torch.tensor([f.label for f in train], dtype=torch.long)
        dev_labels = torch.tensor([f.label for f in dev], dtype=torch.long)
    elif output_mode == "regression":
        train_labels = torch.tensor([f.label for f in train], dtype=torch.float)
        dev_labels = torch.tensor([f.label for f in dev], dtype=torch.float)

    train_dataset = TensorDataset(train_input_ids, train_attention_mask,
                                  train_token_type_ids, train_labels)
    dev_dataset = TensorDataset(dev_input_ids, dev_attention_mask,
                                dev_token_type_ids, dev_labels)

    return train_dataset, dev_dataset, num_labels

def launch(task_params, env_params, model_params,
           optim_params, data_params, trainer_params):
    main_params = {}

    # print params
    if (env_params['distributed'] == False or
        env_params['rank'] == 0):
        print('env_params:\t', env_params)
        print('model_params:\t', model_params)
        print('optim_params:\t', optim_params)
        print('data_params:\t', data_params)
        print('trainer_params:\t', trainer_params)

    # computation env
    set_up_env(env_params)
    device = env_params['device']

    logger = Logger()

    for task in task_params:
        print (task)

        task_config = task_params[task]
        task_config["block_size"] = model_params["block_size"]
        trainer_params["batch_size"] = task_config["batch_size"]

        # model
        model = GenDisc(vocab_size=data_params['vocab_size'],
                        batch_size=trainer_params["batch_size"],
                        model_params=model_params)
        model = model.to(device)

        data_path = data_params["data_path"]
        tokenizer = CharBPETokenizer(join(data_path, "tokenizer-vocab.json"),
                                     join(data_path, "tokenizer-merges.txt"),
                                     unk_token="[UNK]")

        train_data, val_data, num_labels = load_glue(tokenizer, task, task_config)

        # optimizer, scheduler, logger and resume from checkpoint
        optim_params = task_config["optim_params"]
        optimizer, scheduler = get_optimizer_and_scheduler(
            model=model, optim_params=optim_params)

        # reload checkpoint
        main_params["iter_init"] = load_checkpoint(
            trainer_params['checkpoint_path'], trainer_params['last_iter'],
            model, optimizer, scheduler,
            logger, parallel=False)

        asct = AsctSequenceClassification(task_config, model_params, model, num_labels)
        asct = asct.to(device)

        # store main params
        main_params["model"] = asct
        main_params["device"] = device
        main_params["optimizer"] = optimizer
        main_params["scheduler"] = scheduler
        main_params["logger"] = logger
        
        train_glue(train_data, val_data, main_params, trainer_params, env_params, task_config)
        return

if __name__ == '__main__':
    launch(TASK_CONFIG,
           **get_params(params_config=CONFIG))
