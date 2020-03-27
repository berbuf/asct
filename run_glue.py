#!/usr/bin/env python3

import math
import time
from os.path import join, exists

import torch

from config import PARAMS_CONFIG_SMALL as CONFIG
from task_config import PARAMS_CONFIG_TASKS as TASK_CONFIG

from data import get_train_val_test_data
from models import GenDisc
from trainer import train_iteration, full_eval
from utils import (get_params, set_up_env,
                   get_optimizer_and_scheduler,
                   load_checkpoint, save_checkpoint, Logger)

from glue.processors.glue import glue_processors as processors
from glue.processors.glue import glue_output_modes as output_modes
from glue.metrics import glue_compute_metrics as compute_metrics
from glue.processors.glue import glue_convert_examples_to_features as convert_examples_to_features

from tokenizers import CharBPETokenizer

def eval_only(model_params, env_params, main_params,
              val_data, test_data):

    # evaluate the model on test data
    with torch.no_grad():
        loss_val = full_eval(main_params,
                             model_params["block_size"],
                             data=val_data)
        loss_test = full_eval(main_params,
                            model_params["block_size"],
                              data=test_data)

        # collect results
        if env_params['distributed']:
            stats = torch.tensor([loss_val, loss_test]).to(device)
            torch.distributed.reduce(stats, 0)
            if env_params['rank'] != 0:
                return
            loss_val = stats[0] / env_params['world_size']
            loss_test = stats[1] / env_params['world_size']

        # final eval
        print('val: {:.3f}bpc'.format(loss_val / math.log(2)))
        print('test: {:.3f}bpc'.format(loss_test / math.log(2)))

def train_only(model_params, trainer_params,
               env_params, main_params,
               val_data, train_data):

    # iter on epochs
    data_pos = [0] * 2
    for iter_no in range(main_params["iter_init"],
                         trainer_params['nb_iter']):

        # train epoch
        t_sta = time.time()
        loss_train, data_pos[0] = (
            train_iteration(main_params,
                            trainer_params,
                            model_params["block_size"],
                            data=train_data, eval_only=False,
                            train_pos=data_pos[0]))
        elapsed = (1000 * (time.time() - t_sta) /
                   trainer_params["nb_batches_per_iter"])

        # valid epoch
        with torch.no_grad():
            loss_val, data_pos[1] = (
                train_iteration(main_params, trainer_params,
                                model_params["block_size"],
                                data=val_data, eval_only=True,
                                train_pos=data_pos[1]))

        # collect results
        if env_params['distributed']:
            stats = torch.tensor(
                [loss_train, loss_val]).to(main_params["device"])
            torch.distributed.reduce(stats, 0)
            if env_params['rank'] != 0:
                continue
            loss_train = stats[0] / env_params['world_size']
            loss_val = stats[1] / env_params['world_size']

        # checkpoint
        main_params["logger"].log_iter(main_params, trainer_params,
                                       iter_no, elapsed,
                                       loss_train, loss_val)
        save_checkpoint(trainer_params['checkpoint_path'],
                        iter_no, main_params)

def load_glue(tokenizer, task, task_config):
    data_path = task_config["data_path"]

    processor = processors[task]()
    output_mode = output_modes[task]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    file_name_train = "{}_{}_{}".format("train", task_config["max_seq_length"], task)
    file_name_dev = "{}_{}_{}".format("dev", task_config["max_seq_length"], task)

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
            max_length=task_config["max_seq_length"],
            output_mode=output_mode)
        dev = convert_examples_to_features(
            dev, tokenizer, label_list=label_list,
            max_length=task_config["max_seq_length"],
            output_mode=output_mode)

        torch.save(train, train_path)
        torch.save(dev, dev_path)

    return train, dev
        
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

    # model
    model = GenDisc(vocab_size=data_params['vocab_size'],
                    batch_size=trainer_params["batch_size"],
                    model_params=model_params)
    model = model.to(device)

    # distributed
    if env_params['distributed']:
        local_rank = env_params['local_rank']
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
            output_device=local_rank)
    else:
        #model = torch.nn.DataParallel(model)
        pass

    logger = Logger()

    data_path = data_params["data_path"]
    tokenizer = CharBPETokenizer(join(data_path, "tokenizer-vocab.json"),
                                 join(data_path, "tokenizer-merges.txt"),
                                 unk_token="[UNK]")

    for task in task_params:
        print (task)

        task_config = task_params[task]

        train_data, val_data = load_glue(tokenizer, task, task_config)

        # optimizer, scheduler, logger and resume from checkpoint
        optim_params = task_config["optim_params"]
        optimizer, scheduler = get_optimizer_and_scheduler(
            model=model, optim_params=optim_params)

        # reload checkpoint
        path_ckpt = (trainer_params['checkpoint_path']
                     + "_iter_" + str(trainer_params['last_iter']))
        main_params["iter_init"] = load_checkpoint(
            path_ckpt, model,
            optimizer, scheduler,
            logger, env_params['distributed'])

        # store main params
        main_params["model"] = model
        main_params["device"] = device
        main_params["optimizer"] = optimizer
        main_params["scheduler"] = scheduler
        main_params["logger"] = logger

        continue

        train_only(model_params, trainer_params,
                   env_params, main_params,
                   val_data, train_data)

if __name__ == '__main__':
    launch(TASK_CONFIG,
           **get_params(params_config=CONFIG))
