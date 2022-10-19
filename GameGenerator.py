#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:34:25 2022

@author: G.Gradziuk
"""
from Game import *
import copy
import numpy as np

def StateGenerator(n=4, strategy=None, max_depth=100, out_format='state'):
    while(True):
        game = Game(n, strategy=strategy)
        states = []
        moves = []
        weights = []
        while game.winner == 0 and len(game.empty_fields) > 0:
            current_state = copy.deepcopy(game.state)
            states.append(current_state)
            weights.append(game.turn)
            game.make_move()
            moves.append(game.last_field)
            
        if game.winner == 0: # Game ended with a draw, not useful for learning.
            continue
        
        ### crop the early stages ###
        states = np.array(states[-max_depth:], dtype=np.float32)
        weights = np.array(weights[-max_depth:], dtype=np.float32)
        moves = np.array(moves[-max_depth:], dtype=np.int32)
        
        ### IMPORTANT: this facilitates training ###
        ### this way when making a decision, opponent's fields are always -1 ###
        states = states * np.reshape(weights, (-1, 1))
        
        weights *= weights[-1] #winning move weights positive
        weights *= np.linspace(1 / weights.shape[-1], 1, weights.shape[-1])
        ### the closer to the end of the game, the more important the move ###
        ### for example for for a game ended after 5 moves:                ###
        ### overal_weights = array([ 0.2, -0.4,  0.6, -0.8,  1. ])         ###
        
        if out_format == 'state':
            for state, move, weight in zip(states, moves, weights):
                yield state, move, weight
        else: #output the whole game
            yield states, moves, weights
            
def BatchGenerator(n, state_gen, batch_size): #state_gen must output states, not full games
    states = np.zeros((batch_size, n**3), dtype=np.float32)
    moves = np.zeros((batch_size), dtype=np.int32)
    weights = np.zeros((batch_size), dtype=np.float32)
    
    while True:
        for i in range(batch_size):
            states[i], moves[i], weights[i] = next(state_gen)
        yield states, moves, weights
    
#TODO (Done): in the model, make sure that "my" fields are always represented by 1, not -1, to facilitate training
    