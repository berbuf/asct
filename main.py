#!/usr/bin/env python3

import math
import time

import torch

from config import PARAMS_CONFIG_SMALL as CONFIG
from data import get_train_val_test_data
from models import GenDisc
from contextual_loss import ContextualLoss
from trainer import train_iteration, full_eval
from utils import (get_params, set_up_env,
                   get_optimizer_and_scheduler,
                   load_checkpoint, save_checkpoint, Logger)

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
    train_data, val_data, test_data = (
        get_train_val_test_data(data_params, env_params,
                                trainer_params['batch_size'],
                                device))

    # model
    model = GenDisc(vocab_size=data_params['vocab_size'],
                    batch_size=trainer_params["batch_size"],
                    model_params=model_params)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # contextual loss
    #main_params["loss"] = ContextualLoss(data_params["vocab_size"],
    #                                     trainer_params["batch_size"],
    #                                     **model_params)

    # optimizer, scheduler, logger and resume from checkpoint
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model, optim_params=optim_params)
    logger = Logger()
    main_params["iter_init"] = load_checkpoint(
        trainer_params['checkpoint_path'],
        trainer_params['last_iter'], model,
        optimizer, scheduler,
        logger, parallel=True)

    # store main params
    main_params["model"] = model
    main_params["device"] = device
    main_params["optimizer"] = optimizer
    main_params["scheduler"] = scheduler
    main_params["logger"] = logger

    save_checkpoint(checkpoint_path=trainer_params['checkpoint_path'],
                    iter_no=0, main_params=main_params)
    return

    # iter
    if trainer_params['full_eval_mode']:
        eval_only(model_params, trainer_params,
                  env_params, main_params,
                  val_data, test_data)
    else:
        train_only(model_params, trainer_params,
                   env_params, main_params,
                   val_data, train_data)

if __name__ == '__main__':
    launch(**get_params(params_config=CONFIG))
