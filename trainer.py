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
import numpy as np

def train_imdb_vanilla(model_params, trainer_params, model,
                      optimizer, train_data, val_data):
    batch_size = trainer_params["batch_size"]

    model.train()

    nb_batches = train_data[0].shape[0]//batch_size
    for iter_no in range(trainer_params['nb_iter']):
        acc_acc, acc_loss = 0., 0.
        for batch in tqdm(range(nb_batches)):
            batch_data = train_data[0][batch * batch_size: (batch + 1) * batch_size]
            batch_label = train_data[1][batch * batch_size: (batch + 1) * batch_size]            

            optimizer.zero_grad()

            loss, acc = model(batch_data, batch_label)

            loss.backward()
            optimizer.step()

            acc_acc += acc.item()
            acc_loss += loss.item()

        msg = "Step:{:<5}, Acc:{:<10}, L:{:<10}"
        print (msg.format(iter_no,
                          np.round(acc_acc / nb_batches, 4),
                          np.round(acc_loss / nb_batches, 4)))

def step(eval, iter_no, data, model, optimizer, trainer_params):
    batch_size = trainer_params["batch_size"]
    loss_asa = trainer_params["loss_asa"]
    loss_act = trainer_params["loss_act"]

    if not eval:
        model.train()
    else:
        model.eval()

    acc_acc, acc_total_loss, acc_loss, acc_asa, acc_act, acc_updates = 0., 0., 0., 0., 0., 0.
    acc_track_asa, acc_track_act = [], []
    nb_batches = data[0].shape[0]//batch_size
    for batch in tqdm(range(nb_batches)):
    #for batch in range(train_data[0].shape[0]//batch_size):
        batch_data = data[0][batch * batch_size: (batch + 1) * batch_size]
        batch_label = data[1][batch * batch_size: (batch + 1) * batch_size]            

        if not eval:
            optimizer.zero_grad()

        loss, acc = model(batch_data, batch_label)

        track_act = model.model.track_act
        track_asa = model.model.track_asa 
        updates = model.model.act.updates
        norm_asa = model.model.layer.attn.norm_span / updates# * loss_asa
        remainders = model.model.act.get_remainders().mean()# * loss_act

        total_loss = loss + remainders * loss_act + norm_asa * loss_asa
        if not eval:
            total_loss.backward()
            optimizer.step()

        acc_acc += acc.item()
        acc_loss += loss.item()
        acc_total_loss += total_loss.item()
        acc_act += remainders.item()
        acc_asa += norm_asa.item()
        acc_updates += updates
        acc_track_act += [track_act]
        acc_track_asa += [track_asa]

    msg = "Step:{:<3}, Eval:{:<5}, Acc:{:<6}, TL:{:<6}, L:{:<6}, Asa:{:<6}, Act:{:<6}, Updates:{:<5}"
    msg = msg.format(
        iter_no,
        str(eval),
        np.round(acc_acc / nb_batches, 4),
        np.round(acc_total_loss / nb_batches, 4),
        np.round(acc_loss / nb_batches, 4),
        np.round(acc_asa / nb_batches, 4),
        np.round(acc_act / nb_batches, 4),
        np.round(acc_updates / nb_batches, 4))

    m = max([ len(e) for e in acc_track_act ])
    acc_track_act = [ e + [0] * (m - len(e)) for e in acc_track_act ]
    mean_act = np.array(acc_track_act).mean(0)
    act_msg = "act: " + " ".join([ str(np.round(e, 1)) for e in mean_act ])

    acc_track_asa = [ e + [-1] * (m - len(e)) for e in acc_track_asa ]
    mean_asa = np.array(acc_track_asa)
    res_asa = []
    for i in range(mean_asa.shape[1]):
        res_asa += [mean_asa[:,i][mean_asa[:,i] != -1].mean()]
    asa_msg = "asa: " + " ".join([ str(np.round(e, 1)) for e in res_asa ])

    log_path = "./logs/{}.act_{}.asa_{}".format(trainer_params["checkpoint_path"],
                                                str(loss_act), str(loss_asa))
    with open(log_path, "a") as f:
        f.write(msg+"\n")
        f.write(act_msg+"\n")
        f.write(asa_msg+"\n")

    print (msg)
    print (act_msg)
    print (asa_msg)

def train_imdb(model_params, trainer_params, model,
               optimizer, train_data, val_data):
    iter_start = trainer_params["last_iter"]
    if not iter_start:
        log_path = "./logs/{}.act_{}.asa_{}".format(trainer_params["checkpoint_path"],
                                                    str(trainer_params["loss_act"]),
                                                    str(trainer_params["loss_asa"]))
        if os.path.isfile(log_path):
            os.remove(log_path)

    for iter_no in range(iter_start, trainer_params['nb_iter']):
        eval=False
        step(eval, iter_no, train_data, model, optimizer, trainer_params)
        eval=True
        step(eval, iter_no, val_data, model, optimizer, trainer_params)
        save_checkpoint(trainer_params, model, optimizer, iter_no)

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

