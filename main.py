#!/usr/bin/env python3

import math
import time

import torch

from config import PARAMS_CONFIG
from data import get_train_val_test_data
from models import Transformer
from trainer import train_iteration, full_eval
from utils import (
    get_params,
    set_up_env,
    get_optimizer_and_scheduler,
    load_checkpoint,
    save_checkpoint,
    Logger)

from contextual_loss import ContextualLoss

def eval_only(model_params,
              env_params,
              model,
              contextual_loss,
              optimizer,
              scheduler,
              val_data,
              test_data,
              device):
    distributed = env_params['distributed']
    # evaluate the model on test data
    with torch.no_grad():
        loss_val = full_eval(model=model,
                             contextual_loss=contextual_loss,
                             optimizer=optimizer,
                             scheduler=scheduler,
                             data=val_data,
                             block_size=model_params['block_size'],
                             hidden_size=model_params['hidden_size'])
        loss_test = full_eval(model=model,
                              contextual_loss=contextual_loss,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              data=test_data,
                              block_size=model_params['block_size'],
                              hidden_size=model_params['hidden_size'])
        # collect results
        if distributed:
            stats = torch.tensor([loss_val, loss_test]).to(device)
            torch.distributed.reduce(stats, 0)
            if env_params['rank'] == 0:
                loss_val = stats[0] / env_params['world_size']
                loss_test = stats[1] / env_params['world_size']
            else:
                return
        print('val: {:.3f}bpc'.format(loss_val / math.log(2)))
        print('test: {:.3f}bpc'.format(loss_test / math.log(2)))

def train_only(model_params,
               train_params,
               env_params,
               model,
               optimizer,
               scheduler,
               val_data,
               train_data,
               device,
               logger):
    distributed = env_params['distributed']
    # position of current batch
    data_pos = [0] * 2
    # initialize caches for train and valid
    hid_cache = [
        get_cache(train_data, model_params['hidden_size'], model),
        get_cache(train_data, model_params['hidden_size'], model)
    ]
    # iter on epochs
    nb_batches_per_iter = trainer_params['nb_batches_per_iter']
    for iter_no in range(iter_init, trainer_params['nb_iter']):
        # train epoch
        t_sta = time.time()
        loss_train, data_pos[0], hid_cache[0] = train_iteration(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data=train_data,
            nb_batches_per_iter=nb_batches_per_iter,
            block_size=model_params['block_size'],
            eval_only=False,
            train_pos=data_pos[0],
            h_cache=hid_cache[0],
            batch_split=trainer_params['batch_split'])
        elapsed = 1000 * (time.time() - t_sta) / nb_batches_per_iter
        # valid epoch
        with torch.no_grad():
            loss_val, data_pos[1], hid_cache[1] = train_iteration(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                data=val_data,
                nb_batches_per_iter=nb_batches_per_iter,
                block_size=model_params['block_size'],
                eval_only=True,
                train_pos=data_pos[1],
                h_cache=hid_cache[1],
                batch_split=trainer_params['batch_split'])
        # collect results
        if distributed:
            stats = torch.tensor(
                [loss_train, loss_val]).to(device)
            torch.distributed.reduce(stats, 0)
            if env_params['rank'] == 0:
                loss_train = stats[0] / env_params['world_size']
                loss_val = stats[1] / env_params['world_size']
            else:
                continue
        # checkpoint
        logger.log_iter(iter_no, nb_batches_per_iter, loss_train,
                        loss_val, elapsed, model)
        save_checkpoint(trainer_params['checkpoint_path'],
                        iter_no, model, optimizer, scheduler, logger)

def launch(env_params, model_params,
           optim_params, data_params, trainer_params):
    # env (device, distributed, etc.)
    distributed = env_params['distributed']
    set_up_env(env_params)
    device = env_params['device']
    # data
    train_data, val_data, test_data = get_train_val_test_data(
        data_params=data_params,
        env_params=env_params,
        batch_size=trainer_params['batch_size'],
        device=device)
    # print params
    if distributed == False or env_params['rank'] == 0:
        print('model_params:\t', model_params)
        print('optim_params:\t', optim_params)
        print('data_params:\t', data_params)
        print('trainer_params:\t', trainer_params)
    # model
    model = Transformer(
        vocab_size=data_params['vocab_size'], **model_params)
    # distribute
    if distributed:
        local_rank = env_params['local_rank']
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    else:
        #model = torch.nn.DataParallel(model)
        model = model.to(device)
    # optimizer, scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model, optim_params=optim_params)
    # logger
    logger = Logger()
    # resume training from last checkpoint if exists
    iter_init = load_checkpoint(
        trainer_params['checkpoint_path'], model, optimizer, scheduler,
        logger, distributed)
    # contextual loss
    contextual_loss = ContextualLoss(model_params["dup_batch_size"],
                                     model_params["block_size"],
                                     data_params["vocab_size"],
                                     model_params["context_loss_scale"])
    # iter
    if trainer_params['full_eval_mode']:
        eval_only(model_params,
                  env_params,
                  model,
                  contextual_loss,
                  optimizer,
                  scheduler,
                  val_data,
                  test_data,
                  device)
    else:
        train_only(model_params,
                   train_params,
                   env_params,
                   model,
                   optimizer,
                   scheduler,
                   val_data,
                   train_data,
                   device,
                   logger)

if __name__ == '__main__':
    launch(**get_params(params_config=PARAMS_CONFIG))
