#!/usr/bin/env python3

import os
import math
import random
import time
from data import get_data_block, mask
from utils import save_checkpoint
import torch.nn.functional as F
import torch
from tqdm import tqdm

def train_iteration(main_params, trainer_params, block_size,
                    data, mask_idx, eval_only, tokenizer):
    optimizer, scheduler = main_params["optimizer"], main_params["scheduler"]
    model, context_loss = main_params["model"], main_params["loss"]
    batch_size = trainer_params["batch_size"]
    # set model
    main_params["model"].eval() if eval_only else main_params["model"].train()
    # loop parameters
    acc_loss_gen, acc_loss_disc, acc_loss_asa, acc_loss_act, acc_exit, nb_batch = (
        0., 0., 0., 0., 0., 0)
    print (data.shape)
    for batch in tqdm(range(data.shape[1]//block_size)):
        # batch step
        batch_data = data[:, batch * block_size: (batch + 1) * block_size]

        if batch_data.shape[1] != block_size:
            print ("exit")
            break

        mask_data, label_idx = mask(batch_data.clone(), mask_idx)

        if not eval_only:
            optimizer.zero_grad()

        out_gen, out_disc = model(mask_data)

        # loss generator
        _,_,V=out_gen.size()
        total_loss_gen = F.nll_loss(out_gen[:,label_idx].view(-1, V),
                              batch_data[:,label_idx].view(-1))

        # loss discriminator
        gen_data = out_gen.argmax(2)
        label_disc = gen_data == batch_data
        out_disc = out_disc.squeeze()
        loss_disc = (label_disc * (1 - out_disc) + ~label_disc * out_disc).mean()

        loss_act = model.disc.act_module.remainders.mean()
        loss_asa = model.disc.layer.attn.norm_span
        exit_ = model.disc.act_module.exit_.float().mean()

        total_loss_disc = (loss_disc
                           + loss_act * trainer_params["loss_act"]
                           + loss_asa * trainer_params["loss_asa"]
        )

        total_loss = (total_loss_gen * trainer_params["loss_gen"]
                      + total_loss_disc
        )

        if not eval_only:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.) 
            if scheduler is not None:
                scheduler.step()
            optimizer.step()

        # loop parameters
        acc_loss_gen += total_loss_gen.item()
        acc_loss_disc += loss_disc.item()
        acc_loss_act += loss_act.item()
        acc_loss_asa += loss_asa.item()
        acc_exit += exit_
        nb_batch += 1

    return (
        acc_loss_gen / nb_batch,
        acc_loss_disc / nb_batch,
        acc_loss_act / nb_batch,
        acc_loss_asa / nb_batch,
        acc_exit / nb_batch,
    )

def train_only(model_params, trainer_params,
               env_params, main_params, data_path,
               device, tokenizer):
    mask_idx = tokenizer.token_to_id("[MASK]")
    # iter on epochs
    for iter_no in range(main_params["iter_init"],
                         trainer_params['nb_iter']):

        for file in os.listdir(data_path):
            if (file.find("uncased_chunk") == -1
                or file.find("corpus.pt") != -1) :
                continue

            # data_block
            train_data = get_data_block(data_path, trainer_params["batch_size"],
                                        device, tokenizer, file)

            # train epoch
            t_sta = time.time()
            loss_gen, loss_disc, loss_act, loss_asa, exit_ = (
                train_iteration(main_params,
                                trainer_params,
                                model_params["block_size"],
                                data=train_data,
                                mask_idx=mask_idx,
                                eval_only=False,
                                tokenizer=tokenizer))

            print ("gen:{:<20}, disc:{:<20}, act:{:<20}, asa:{:<20}, exit:{:<20}".format(
                loss_gen, loss_disc, loss_act, loss_asa, exit_))

            elapsed = (1000 * (time.time() - t_sta) /
                       trainer_params["nb_batches_per_iter"])

            # checkpoint
            main_params["logger"].log_iter(main_params, trainer_params,
                                           iter_no, file, elapsed,
                                           loss_disc)

            save_checkpoint(trainer_params, main_params, iter_no)

