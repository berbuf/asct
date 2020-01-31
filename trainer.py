#!/usr/bin/env python3

import math
import random
import torch

def _train_step(model, X, Y, eval_only, loss_div=1):
    # forward
    out = model(X)
    # compute loss
    out = out.view(-1, out.size(-1))
    loss = torch.nn.functional.nll_loss(out, Y.view(-1))
    loss_value = loss.item() / loss_div
    if not eval_only:
        # compute loss for adaptive span
        if model.module.layers[0].attn.attn.adapt_span_enabled:
            loss += sum(layer.attn.attn.adaptive_span.get_loss()
                        for layer in model.module.layers)
        # backpropagate loss
        (loss / loss_div).backward()
    return loss_value

def _train_batch(model, optimizer, scheduler, X, Y,
                 eval_only, batch_split):
    # start gradient
    if not eval_only:
        optimizer.zero_grad()
    # train step
    loss_value = _train_step(model, X, Y, eval_only)
    if not eval_only:
        # schedule lr
        if scheduler is not None:
            scheduler.step()
        # gradient descent
        optimizer.step()
        # clamp adaptive span size
        if model.module.layers[0].attn.attn.adapt_span_enabled:
            for layer in model.module.layers:
                layer.attn.attn.adaptive_span.clamp_param()
    return loss_value

def train_iteration(model, optimizer, scheduler, data, nb_batches_per_iter,
                    block_size, eval_only, train_pos, h_cache, batch_split):
    # set model for train
    model.eval() if eval_only else model.train()
    # less batch for speed-up 
    if eval_only:
        nb_batches_per_iter = max(1, nb_batches_per_iter // 10)
        nb_batches_per_iter = min(nb_batches_per_iter,
                                  math.ceil(data.size(1) / block_size))
    # loop parameters
    loss_all = 0
    actual_nb_batches_per_iter = 0
    for _ in range(nb_batches_per_iter):
        # batch data 
        X = data[:, train_pos: train_pos + block_size].contiguous()
        Y = data[:, train_pos + 1: train_pos + block_size + 1].contiguous()
        # batch step
        loss, h_cache = _train_batch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            X=X, Y=Y,
            h_cache=h_cache,
            eval_only=eval_only,
            batch_split=batch_split)
        # loop parameters
        loss_all += loss
        train_pos += block_size
        actual_nb_batches_per_iter += 1
        # reached the end. randomize the offset to reduce overfitting
        if train_pos >= data.size(1) - block_size:
            train_pos = random.randrange(block_size)
            # reset the cache
            for h in h_cache:
                h.fill_(0)
    return (loss_all / actual_nb_batches_per_iter,
            train_pos, h_cache)

def full_eval(model, optimizer, scheduler, data, block_size, hidden_size):
    # eval mode
    model.eval()
    # loop parameters
    loss_all = 0
    train_pos = 0
    actual_nb_batches_per_iter = 0
    # iter on batches
    nb_batches_per_iter_max = math.ceil(data.size(1) / block_size)
    for _ in range(nb_batches_per_iter_max):
        # batch data 
        X = data[:, train_pos: train_pos + block_size].contiguous()
        Y = data[:, train_pos + 1: train_pos + block_size + 1].contiguous()
        # batch step
        loss = _train_batch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            X=X, Y=Y,
            eval_only=True,
            batch_split=1)
        # loop parameters
        loss_all += loss
        train_pos += block_size
        actual_nb_batches_per_iter += 1
        if train_pos >= data.size(1) - block_size:
            # Skip the remaining tokens as it can't make a whole block.
            break
    return loss_all / actual_nb_batches_per_iter
