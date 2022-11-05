#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:58:26 2022

@author: G.Gradziuk
"""

from Game import *
from GameGenerator import *
# from Reports import *
from SavingModule import *

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

n = 3
epsilon = 1e-8
np.random.seed(42)

init_model_name = ""

### Training parameters ###
batch_size = 300
n_batches = 100
n_epochs = 201
max_depth = 100
temperature = 0.1
rand_move_rate = 0.15
weight_profile = "flat"
train_on_random = False

bad_penalty_factor = 1

additional_notes = "Trying the new accuracy metrics."
#TODO: try randomizing only one player for generating data
#TODO: after training with maxdepth=100 it learns how to end games quickly
#      and so it works really bad for advanced game stages
#   -> maybe train on a mixture of games, where each game starts from a state arrived at randomly (x random moves not used for training, followed by a reasonable moves used for training)


# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(100, activation=tf.nn.relu, input_shape=(n**3,)),
#     tf.keras.layers.Dense(100, activation=tf.nn.relu),
#     tf.keras.layers.Dense(n**3, activation=tf.nn.relu),
#     tf.keras.layers.Softmax()
#     ])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(60, activation=tf.nn.relu, input_shape=(n**3,)),
    tf.keras.layers.Dense(60, activation=tf.nn.relu),
    tf.keras.layers.Dense(60, activation=tf.nn.relu),
    tf.keras.layers.Dense(n**3, activation=tf.nn.relu),
    tf.keras.layers.Softmax()
    ])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

init_model_name = "models/Adam_0.01/max_depth=100/temp=0.1_rand_move_rate=0.15/27-10-2022_20_08_31"
model = tf.keras.models.load_model(init_model_name)

data_gen_model_name = "models/Adam_0.01/max_depth=100/temp=0.1_rand_move_rate=0.15/27-10-2022_20_08_31"
data_gen_model = tf.keras.models.load_model(data_gen_model_name)



def loss_fun(model, states, moves, weights, training=False):
    move_probs = model(states, training=training)
    
    ### loss1 -> penalty for choosing occupied fields ###
    is_occupied = tf.math.abs(states)
    abs_weights = tf.math.abs(weights)
    loss1 = is_occupied * tf.math.log(1 - move_probs + epsilon)
    loss1 = tf.reduce_sum(loss1, axis=-1) / (tf.reduce_sum(is_occupied, axis=-1) + epsilon)
    # loss1 = - tf.tensordot(abs_weights, loss1, 1)
    loss1 = - tf.reduce_sum(loss1, axis=-1)
    # TODO(Done): Actually, loss1 should not include weights, forbidden moves are always clearly bad...
    
    ### loss2 -> promote good / penalize bad moves
    prob_for_made_move = tf.gather(move_probs, moves, axis=-1, batch_dims=1) 
    loss2 = tf.where(tf.less(0, weights), prob_for_made_move, (1 - prob_for_made_move) )
    loss2 = tf.math.log(loss2 + epsilon)
    loss2 = tf.where(tf.less(0, weights), loss2, loss2 * bad_penalty_factor )
    loss2 = - tf.tensordot(abs_weights, loss2, 1)
    
    return (loss1 + loss2) / weights.shape[0]
#TODO (done): Check if loss scale depends on the batch size

def grad(model, states, moves, weights):
    with tf.GradientTape() as tape:
        loss = loss_fun(model, states, moves, weights, training=True)
    return loss, tape.gradient(loss, model.trainable_variables)

def get_metrics(model, states, moves, weights, training=False):
    move_probs = model(states, training=training)
    forbidden_probs = move_probs * tf.math.abs(states)
    tot_p_forbidden = tf.reduce_sum(forbidden_probs, axis=-1)
    mean_tot_p_forbidden = tf.reduce_mean(tot_p_forbidden)
    
    p_made_move = tf.gather(move_probs, moves, axis=-1, batch_dims=1) 
    good_move_prob = tf.gather(p_made_move, tf.where(weights > 0))
    bad_move_prob = tf.gather(p_made_move, tf.where(weights < 0))
    mean_good_prob = tf.reduce_mean(good_move_prob)
    mean_bad_prob = tf.reduce_mean(bad_move_prob)
    
    #TODO: add metrics: %good move chosen, %bad move chosen, %allowed move chosen
    # additional metrics:
    allowed_move_probs = tf.where(tf.equal(0, states), move_probs, 0)
    allowed_guessed_moves = tf.argmax(allowed_move_probs, axis=-1)
    guessed_moves = tf.argmax(move_probs, axis=-1)
    
    allowed_guessed_made_overlap = tf.cast(tf.equal(allowed_guessed_moves, moves), dtype=tf.float32)
    is_allowed_move_guessed = tf.cast(tf.equal(allowed_guessed_moves, guessed_moves), dtype=tf.float32)
    
    bad_move_rate = tf.reduce_mean( tf.gather(allowed_guessed_made_overlap, tf.where(weights < 0)) )
    good_move_rate = tf.reduce_mean( tf.gather(allowed_guessed_made_overlap, tf.where(weights > 0)) )
    allowed_move_rate = tf.reduce_mean(is_allowed_move_guessed)
    
    return mean_tot_p_forbidden, mean_good_prob, mean_bad_prob, good_move_rate, bad_move_rate, allowed_move_rate



loss_l = []
p_forbidden_l = []
p_good_l = []
p_bad_l = []
good_rate_l = []
bad_rate_l = []
allowed_rate_l = []

saving_module = SavingModule(model=model,
                            optimizer=optimizer,
                            max_depth=max_depth,
                            temperature=temperature,
                            rand_move_rate=rand_move_rate,
                            weight_profile=weight_profile,
                            train_on_random=train_on_random,
                            
                            batch_size=batch_size,
                            n_batches=n_batches,
                            n_epochs=n_epochs,
                            init_model_name=init_model_name,
                            data_gen_model_name=data_gen_model_name,
                            additional_notes=additional_notes)

if train_on_random:
    state_gen = StateGenerator(n, max_depth=max_depth, weight_profile=weight_profile)
    temperature, rand_move_rate = 0, 0
    data_gen_model_name = ""
else:
    state_gen = StateGenerator(n, max_depth=max_depth,
                               strategy=model,
                               rand_move_rate=rand_move_rate,
                               temperature=temperature,
                               weight_profile=weight_profile) # tried without added randomness, gets stuck in a shitty fixed point

batch_gen = BatchGenerator(n, state_gen, batch_size)

for epoch in range(n_epochs):
    loss_sum = 0
    for b in range(n_batches):
        data = next(batch_gen)
        loss, grads = grad(model, *data)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        loss_sum += loss
    
    # calculate the metrics for a single batch
    data = next(batch_gen)
    p_forbidden, p_good, p_bad, good_rate, bad_rate, allowed_rate = accuracy(model, *data, training=False)
    p_forbidden_l.append(p_forbidden)
    p_good_l.append(p_good)
    p_bad_l.append(p_bad)
    good_rate_l.append(good_rate)
    bad_rate_l.append(bad_rate)
    allowed_rate_l.append(allowed_rate)
    #TODO: make a dictionary for all these metrics, so that they dont get mixed up by confusing positions?
            
    loss_l.append(loss_sum / n_batches)
    
    if epoch % 10 == 0:
        saving_module.save_checkpoint(model,
                                      epoch=epoch,
                                      loss_l=loss_l,
                                      p_forbidden_l=p_forbidden_l,
                                      p_good_l=p_good_l,
                                      p_bad_l=p_bad_l,
                                      good_rate_l=good_rate_l,
                                      bad_rate_l=bad_rate_l,
                                      allowed_rate_l=allowed_rate_l)
        
        print("After ", epoch,
              " epochs, loss: ", loss_l[-1].numpy(),
              " forbidden: ", np.mean(p_forbidden_l[-10:]),
              " p_good: ", np.mean(p_good_l[-10:]),
              " p_bad ", np.mean(p_bad_l[-10:]),
              " good_rate: ", np.mean(good_rate_l[-10:]),
              " bad_rate ", np.mean(bad_rate_l[-10:]),
              " allowed_rate ", np.mean(allowed_rate_l[-10:]))

#TODO: trenuj progresywnie: 2 ostatnie ruchy, 4 ostatnei ruchy, 6 ostatnich ruchów itd.

os.system('say "The training is done! Hurray!"')








