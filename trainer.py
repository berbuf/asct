#!/usr/bin/env python3

import math
import random
import torch

def _train_step(main_params, X, eval_only, loss_div=1):
    return 1
    model, context_loss = main_params["model"], main_params["loss"]
    # forward
    out = model(X)
    # compute loss
    exit_token = model.layer.act.exit_
    loss = contextual_loss.loss(out, Y, exit_token)
    loss_value = loss.item() / loss_div
    #out = out.view(-1, out.size(-1))
    #loss = torch.nn.functional.nll_loss(out, Y.view(-1))
    # backpropagation
    if not eval_only:
        (loss / loss_div).backward()
    return loss_value

def _train_batch(main_params, X, eval_only, batch_split):
    optimizer, scheduler = main_params["optimizer"], main_params["scheduler"]
    # clear gradient
    if not eval_only:
        optimizer.zero_grad()
    loss_value = _train_step(main_params, X, eval_only)
    # gradient descent
    if not eval_only:         
        if scheduler is not None:
            scheduler.step()
        optimizer.step()
    return loss_value

def train_iteration(main_params, trainer_params, block_size,
                    data, eval_only, train_pos):
    # set model for train
    main_params["model"].eval() if eval_only else main_params["model"].train()
    # less batch for speed-up
    """
    if eval_only:
        nb_batches_per_iter = max(1, nb_batches_per_iter // 10)
        nb_batches_per_iter = min(nb_batches_per_iter,
                                  math.ceil(data.size(1) / block_size))
    """
    # loop parameters
    loss_all = 0
    for _ in range(trainer_params["nb_batches_per_iter"]):
        # batch step
        X = data[:, train_pos: train_pos + block_size].contiguous()
        loss = _train_batch(main_params, X=X, eval_only=eval_only, batch_split=1)
        # loop parameters
        loss_all, train_pos = loss_all + loss, train_pos + block_size
        # reached the end. randomize the offset to reduce overfitting
        if train_pos >= data.size(1) - block_size:
            train_pos = random.randrange(block_size)
    return (loss_all / trainer_params["nb_batches_per_iter"], train_pos)

def full_eval(main_params, block_size, data):
    # eval mode
    main_params["model"].eval()
    # loop parameters
    loss_all, train_pos, nb_batches_per_iter = 0, 0, 0
    for _ in range(math.ceil(data.size(1) / block_size)):
        # batch step
        X = data[:, train_pos: train_pos + block_size].contiguous()
        loss = _train_batch(main_params, X=X, eval_only=True, batch_split=1)
        # loop parameters
        loss_all += loss
        train_pos += block_size
        nb_batches_per_iter += 1
        # Skip the remaining tokens as it can't make a whole block.
        if train_pos >= data.size(1) - block_size:
            break
    return loss_all / nb_batches_per_iter
