#!/usr/bin/env python3

import os
import math
import argparse

import torch
from adagrad_with_grad_clip import AdagradWithGradClip

##############################################################################
# UTILITIES
##############################################################################

def get_r2(M):
    r2, e = [0], 0
    while 2**e <= M:
        e += 1
        # sorted power of 2
        r2 += [2**e]
    return torch.tensor(r2)

def round_r2(r2, v):
    # round to power of 2
    return r2[(r2 < v).sum()] 

##############################################################################
# ARGS
##############################################################################

def _parse_args(params_config, args):
    parser = argparse.ArgumentParser()
    for params_category in params_config:  # e.g., 'model_params'
        for param_flag, param_config in params_config[params_category].items():
            # e.g., param_flag = '--block-sz'
            parser.add_argument(param_flag, **param_config)
    return parser.parse_args(args)

def get_params(params_config, args=None):
    namespace = _parse_args(params_config, args)
    return {
        params_category: {
            param_config['dest']:
                namespace.__getattribute__(param_config['dest'])
            for param_config in params_config[params_category].values()
        }
        for params_category in params_config
    }

##############################################################################
# ENVIRONMENT
##############################################################################

def _torch_distributed_init_process_group(local_rank):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print('my rank={} local_rank={}'.format(rank, local_rank))
    torch.cuda.set_device(local_rank)
    return {
        'rank': rank,
        'world_size': world_size,
    }

def set_up_env(env_params):
    assert torch.cuda.is_available()
    torch.cuda.set_device(env_params['device_num'])
    env_params['device'] = torch.device('cuda')

##############################################################################
# OPTIMIZER AND SCHEDULER
##############################################################################

def _get_grad_requiring_params(model):
    nb_parameters = 0
    grad_requiring_params = []
    for param in model.parameters():
        if param.requires_grad:
            nb_parameters += param.numel()
            grad_requiring_params.append(param)
    print('nb_parameters={:.2f}M'.format(nb_parameters / 1e6))
    return grad_requiring_params

def _get_optimizer(model,
                   optim,
                   lr: float,
                   momentum: float,
                   grad_clip: float):
    if optim == 'sgd':
        return torch.optim.SGD(_get_grad_requiring_params(model),
                               lr=lr,
                               momentum=momentum)
    elif optim == 'adagrad':
        return AdagradWithGradClip(_get_grad_requiring_params(model),
                                   lr=lr,
                                   grad_clip=grad_clip)
    else:
        raise RuntimeError("wrong type of optimizer "
                           "- must be 'sgd' or 'adagrad")

def _get_scheduler(optimizer, lr_warmup):
    if lr_warmup > 0:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: min(1, ep / lr_warmup))
    return None

def get_optimizer_and_scheduler(model, optim_params):
    optimizer = _get_optimizer(model=model,
                               optim=optim_params['optim'],
                               lr=optim_params['lr'],
                               momentum=optim_params['momentum'],
                               grad_clip=optim_params['grad_clip'])
    scheduler = _get_scheduler(optimizer=optimizer,
                               lr_warmup=optim_params['lr_warmup'])
    return optimizer, scheduler

##############################################################################
# CHECKPOINT
##############################################################################

def _load_checkpoint(checkpoint_path, model, optimizer, distributed):
    print('loading from a checkpoint at {}'.format(checkpoint_path))
    if distributed:
        # the model is saved from gpu0 so we need to map it to CPU first
        checkpoint_state = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint_state = torch.load(checkpoint_path)
    iter_init = checkpoint_state['iter_no'] + 1  # next iteration
    model.load_state_dict(checkpoint_state['model'])
    optimizer.load_state_dict(checkpoint_state['optimizer'])

def load_checkpoint(trainer_params, iter_no, model,
                    optimizer, parallel):
    checkpoint_path = trainer_params["checkpoint_path"]
    if checkpoint_path and os.path.exists(checkpoint_path):
        path = "./checkpoint/{}.step_{}.act_{}.asa_{}.{}".format(checkpoint_path,
                                                                 str(iter_no),
                                                                 trainer_params["loss_act"],
                                                                 trainer_params["loss_asa"],
                                                                 "p" if f else "np")
        _load_checkpoint(checkpoint_path=name,
                         model=model,
                         optimizer=optimizer,
                         distributed=distributed)

def save_checkpoint(trainer_params, model, optimizer, iter_no):
    checkpoint_path = trainer_params["checkpoint_path"]
    def parallel(f):
        state_model = model.module if f else model
        checkpoint_state = {
            'iter_no': iter_no,  # last completed iteration
            'model': state_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        path = "./checkpoint/{}.step_{}.act_{}.asa_{}.{}".format(checkpoint_path,
                                                                 str(iter_no),
                                                                 trainer_params["loss_act"],
                                                                 trainer_params["loss_asa"],
                                                                 "p" if f else "np")
        torch.save(checkpoint_state, path)
        print ("Saved", path)
    if checkpoint_path:
        #parallel(f=True)
        parallel(f=False)

##############################################################################
# RAM
##############################################################################

def check_ram(device):
    t = torch.cuda.get_device_properties(device).total_memory
    c = torch.cuda.memory_cached(device)
    a = torch.cuda.memory_allocated(device)
    print ()
    print ("total mem", t)
    print ("total cache", c)
    print ("total allocated", a)

##############################################################################
# LOGGER
##############################################################################

class Logger:
    def __init__(self):
        self._state_dict = dict()

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict

    def _log(self, title, value):
        if title not in self._state_dict:
            self._state_dict[title] = []
        self._state_dict[title].append(value)

    def log_iter(self, main_params, trainer_params,
                 iter_no, file, elapsed,
                 loss):
        train_bpc = float(loss / math.log(2))
        msg = 'iter: {}'.format(iter_no)
        msg += '\ttrain: {:.3f}bpc'.format(train_bpc)
        msg += '\tms/batch: {:.1f}'.format(elapsed)
        self._log(title='iter', value=iter_no)
        self._log(title='file', value=file)
        self._log(title='train_bpc', value=train_bpc)

        print(msg)
