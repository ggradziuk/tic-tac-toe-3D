#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:58:26 2022

@author: G.Gradziuk
"""

from Game import *
from GameGenerator import *
import numpy as np
import tensorflow as tf

n = 3
epsilon = 1e-8

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation=tf.nn.relu, input_shape=(n**3,)),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(n**3, activation=tf.nn.relu),
    tf.keras.layers.Softmax()
    ])

def loss_fun(model, states, moves, weights, training=False):
    move_probs = model(states, training=training)
    
    ### loss1 -> penalty for choosing occupied fields ###
    is_occupied = tf.math.abs(states)
    abs_weights = tf.math.abs(weights)
    loss1 = is_occupied * tf.math.log(1 - move_probs + epsilon)
    loss1 = tf.reduce_sum(loss1, axis=-1) / (tf.reduce_sum(is_occupied, axis=-1) + epsilon)
    loss1 = - tf.tensordot(abs_weights, loss1, 1)
    
    ### loss2 -> promote good / penalize bad moves
    prob_for_made_move = tf.gather(move_probs, moves, axis=-1, batch_dims=1) 
    loss2 = tf.where(tf.less(0, weights), prob_for_made_move, 1 - prob_for_made_move)
    loss2 = tf.math.log(loss2 + epsilon)
    loss2 = - tf.tensordot(abs_weights, loss2, 1)
    
    return (loss1 + loss2) / weights.shape[0]
#TODO (done): Check if loss scale depends on the batch size

def grad(model, states, moves, weights):
    with tf.GradientTape() as tape:
        loss = loss_fun(model, states, moves, weights, training=True)
    return loss, tape.gradient(loss, model.trainable_variables)

optimizer = tf.keras.optimizers.Adagrad(learning_rate=1)

batch_size = 1000
n_batches = 10
n_epochs = 201
max_depth = 4

mean_loss_list = []
state_gen = StateGenerator(n, max_depth=max_depth)
batch_gen = BatchGenerator(n, state_gen, batch_size)

for epoch in range(n_epochs):
    loss_sum = 0
    for _ in range(n_batches):
        data = next(batch_gen)
        loss, grads = grad(model, *data)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        loss_sum += loss
    
    mean_loss_list.append(loss_sum / n_batches)
    if epoch % 10 == 0:
        print("After ", epoch, " epochs, loss: ", mean_loss_list[-1].numpy())



















