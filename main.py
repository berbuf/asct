#!/usr/bin/env python3

import math
import time

import torch

from config import PARAMS_CONFIG_SMALL as CONFIG
from data import get_tokenizer, get_data_imdb
from models import GenDisc, AsctImdbClassification
from contextual_loss import ContextualLoss
from trainer import train_only, train_imdb, train_imdb_vanilla
from utils import (get_params, set_up_env,
                   get_optimizer_and_scheduler,
                   load_checkpoint, save_checkpoint, Logger)

def launch(env_params, model_params,
           optim_params, data_params, trainer_params):
    main_params = {}

    # print params
    print('env_params:\t', env_params)
    print('model_params:\t', model_params)
    print('optim_params:\t', optim_params)
    print('data_params:\t', data_params)
    print('trainer_params:\t', trainer_params)

    # computation env
    set_up_env(env_params)
    device = env_params['device']

    # data
    weights_embed, train_data, val_data, tokenizer = (
        get_data_imdb(trainer_params['batch_size'], device,
                      entry_size=model_params["block_size"],
                      vanilla=model_params["vanilla"], load=False))

    #tokenizer = get_tokenizer(**data_params)

    # model
    #if model_params["vanilla"]:
    #    model = Vanilla()
    model = AsctImdbClassification(
        batch_size=trainer_params["batch_size"],
        weights_embed=weights_embed,
        pad_idx=tokenizer["[PAD]"],
        model_params=model_params,
        vanilla=model_params["vanilla"])
    model = model.to(device)
    #model = torch.nn.DataParallel(model)

    # optimizer, scheduler, logger and resume from checkpoint
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model, optim_params=optim_params)
    #logger = Logger()
    #main_params["iter_init"] = load_checkpoint(
    #    trainer_params,
    #    trainer_params['last_iter'], model,
    #    optimizer, scheduler,
    #    logger, parallel=True)

    # store main params
    #main_params["model"] = model
    #main_params["device"] = device
    #main_params["optimizer"] = optimizer
    #main_params["scheduler"] = scheduler
    #main_params["logger"] = logger

    iter_start = trainer_params["last_iter"]
    if iter_start:
        load_checkpoint(trainer_params, iter_start, model, optimizer, False)

    # iter
    if model_params["vanilla"]:
        train_imdb_vanilla(model_params, trainer_params, model,
                           optimizer, train_data, val_data)
    else:
        train_imdb(model_params, trainer_params, model,
                   optimizer, train_data, val_data)

    #train_only(model_params, trainer_params,
    #           env_params, main_params,
    #           data_params["data_path"],
    #           device, tokenizer)

if __name__ == '__main__':
    launch(**get_params(params_config=CONFIG))
