#!/usr/bin/env python3

import math
import time

import torch

from config import PARAMS_CONFIG_SMALL as CONFIG
from data import get_tokenizer
from models import GenDisc
from contextual_loss import ContextualLoss
from trainer import train_only
from utils import (get_params, set_up_env,
                   get_optimizer_and_scheduler,
                   load_checkpoint, save_checkpoint, Logger)

def launch(env_params, model_params,
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

    # data
    #train_data, val_data, test_data, tokenizer = (
    #    get_train_val_test_data(data_params, env_params,
    #                            trainer_params['batch_size'],
    #                            device))

    tokenizer = get_tokenizer(**data_params)
    pad_idx = tokenizer.token_to_id("[PAD]")

    # model
    model = GenDisc(vocab_size=data_params['vocab_size'],
                    batch_size=trainer_params["batch_size"],
                    model_params=model_params, pad_idx=pad_idx)
    model = model.to(device)
    #model = torch.nn.DataParallel(model)

    # contextual loss
    #main_params["loss"] = ContextualLoss(data_params["vocab_size"],
    #                                     trainer_params["batch_size"],
    #                                     **model_params)
    main_params["loss"] = None

    # optimizer, scheduler, logger and resume from checkpoint
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model, optim_params=optim_params)
    logger = Logger()
    main_params["iter_init"] = load_checkpoint(
        trainer_params,
        trainer_params['last_iter'], model,
        optimizer, scheduler,
        logger, parallel=True)

    # store main params
    main_params["model"] = model
    main_params["device"] = device
    main_params["optimizer"] = optimizer
    main_params["scheduler"] = scheduler
    main_params["logger"] = logger

    #save_checkpoint(checkpoint_path=trainer_params['checkpoint_path'],
    #                iter_no=0, main_params=main_params)

    # iter
    train_only(model_params, trainer_params,
               env_params, main_params,
               data_params["data_path"],
               device, tokenizer)

if __name__ == '__main__':
    launch(**get_params(params_config=CONFIG))
