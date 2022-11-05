#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:31:01 2022

@author: G.Gradziuk
"""

import os
import sys
import datetime
import matplotlib.pyplot as plt
import numpy as np

def PrintReport(model=None,
                optimizer=None,
                loss_l = [],
                p_forbidden_l = [],
                p_good_l = [],
                p_bad_l = [],
                max_depth=0,
                temperature=0,
                rand_move_rate=0,
                weight_profile="",
                train_on_random=True,
                
                batch_size=0,
                n_batches=0,
                n_epochs=0,
                init_model_name="",
                data_gen_model_name="",
                additional_notes=""):
    
    model_name = f"{optimizer._name}_{str(optimizer.learning_rate.numpy())}/"
    model_name += f"max_depth={max_depth}/"
    model_name += f"temp={temperature}_rand_move_rate={rand_move_rate}/"
    model_name += datetime.datetime.now().strftime("%d-%m-%Y_%H_%M_%S/")
    
    model.save("models/" + model_name)
    
    log_path = "logs/" + model_name
    if not os.path.exists(log_path):
       os.makedirs(log_path)
    
    ### generate and save plots ###
    plt.plot(loss_l, '-', label="loss")
    plt.legend()
    plt.savefig(log_path + "loss.png")
    
    plt.clf()
    plt.plot(p_forbidden_l, '-', label="p_forbidden")
    plt.plot(p_good_l, '-', label="p_good")
    plt.plot(p_bad_l, '-', label="p_bad")
    plt.legend()
    plt.savefig(log_path + "probs.png")
    
    ### save numerical data ###
    np.save(log_path + "loss.npy", np.array(loss_l))
    np.save(log_path + "forbidden_prob.npy", np.array(p_forbidden_l))
    np.save(log_path + "good_probs.npy", np.array(p_good_l))
    np.save(log_path + "bad_probs.npy", np.array(p_bad_l))
    
    ### save a detailed log ###
    original_stdout = sys.stdout
    with open(log_path + 'log.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        model.summary()
        print("\n\n")
        print(f"max_depth: {max_depth}")
        print(f"temperature: {temperature}")
        print(f"rand_move_rate: {rand_move_rate}")
        print(f"weight_profile: {weight_profile}")
        print(f"train_on_random: {train_on_random}")
        print("\n")
        print(f"batch_size: {batch_size}")
        print(f"n_batches: {n_batches}")
        print(f"n_epochs: {n_epochs}")
        print("\n")
        print(f"init_model_name: {init_model_name}")
        print(f"data_gen_model_name: {data_gen_model_name}")
        print(f"optimizer: {optimizer._name}_{str(optimizer.learning_rate.numpy())}")
        print(f"additional_notes: {additional_notes}")
        print("\n")
        sys.stdout = original_stdout
    