#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 19:49:15 2022

@author: G.Gradziuk
"""
import os
import sys
import datetime
import matplotlib.pyplot as plt
import numpy as np

class SavingModule:
    def __init__(self,
                 model=None,
                 optimizer=None,
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
        
        self.dir = f"models/{optimizer._name}_{str(optimizer.learning_rate.numpy())}/"
        self.dir += f"max_depth={max_depth}/"
        self.dir += f"temp={temperature}_rand_move_rate={rand_move_rate}/"
        self.dir += datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S/")
        
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        
        ### save a detailed log with training parameters ###
        original_stdout = sys.stdout
        with open(self.dir + 'log.txt', 'w') as f:
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
    
    
    def save_checkpoint(self,
                        model,
                        epoch = None,
                        loss_l = [],
                        p_forbidden_l = [],
                        p_good_l = [],
                        p_bad_l = [],
                        good_rate_l = [],
                        bad_rate_l = [],
                        allowed_rate_l = []):
        
        ### save a checkpoint of the model ###
        model.save(self.dir + f"checkpoints/{epoch}")
        
        ### generate and save plots ###
        plt.plot(loss_l, '-', label="loss")
        plt.legend()
        plt.savefig(self.dir + "loss.png")
        plt.clf()
        
        plt.plot(p_forbidden_l, '-', label="p_forbidden")
        plt.plot(p_good_l, '-', label="p_good")
        plt.plot(p_bad_l, '-', label="p_bad")
        plt.legend()
        plt.savefig(self.dir + "probs.png")
        plt.clf()
        
        plt.plot(good_rate_l, '-', label="good_rate")
        plt.plot(bad_rate_l, '-', label="bad_rate")
        plt.plot(allowed_rate_l, '-', label="allowed_rate")
        plt.legend()
        plt.savefig(self.dir + "rates.png")
        plt.clf()
        
        ### save numerical data ###
        data_dir = self.dir + "data/"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        np.save(data_dir + "loss.npy", np.array(loss_l))
        np.save(data_dir + "p_forbidden.npy", np.array(p_forbidden_l))
        np.save(data_dir + "p_good.npy", np.array(p_good_l))
        np.save(data_dir + "p_bad.npy", np.array(p_bad_l))
        
        np.save(data_dir + "good_rate.npy", np.array(good_rate_l))
        np.save(data_dir + "bad_rate.npy", np.array(bad_rate_l))
        np.save(data_dir + "allowed_rate.npy", np.array(bad_rate_l))
        
        