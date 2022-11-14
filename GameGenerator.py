#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:34:25 2022

@author: G.Gradziuk
"""
from Game import *
import copy
import numpy as np

def StateGenerator(n=4,
                   strategy=None,
                   rand_move_rate=0,
                   temperature=0,
                   weight_profile="flat",
                   max_depth=100):
    """Generator of game states and corrensponding moves and weights for the training.
    
    n - linear dimension of the board (int)
    strategy - function/model: [state] -> [[probabilities for next move]], (default None -> random move)
    rand_move_rate - rate at which a random move is made instead of using the strategy (default 0)
    temperature - stdev of fluctuations added to probabilities returned by the strategy (default 0)
    weight_profile - regulate the importance of final moves compared to initial moves
    max_depth - the number of last moves of each game to yielded. (eg. max_dep=2 -> yield only the last two moves of each game)
    """
    game = Game(n, strategy=strategy, rand_move_rate=rand_move_rate, temperature=temperature)
    while(True):
        game.reset()
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
        
        #TODO: maybe don't throw out the draws
        #####  maybe include them with an appropriate weight
        
        ### crop the early stages ###
        states = np.array(states[-max_depth:], dtype=np.float32)
        weights = np.array(weights[-max_depth:], dtype=np.float32)
        moves = np.array(moves[-max_depth:], dtype=np.int32)
        
        ### IMPORTANT: this facilitates training ###
        ### this way when making a decision, opponent's fields are always -1 ###
        states = states * np.reshape(weights, (-1, 1))
        
        weights *= weights[-1] #winning move weights positive
        if weight_profile == "linear":
            weights *= np.linspace(1 / weights.shape[-1], 1, weights.shape[-1])
        ### the closer to the end of the game, the more important the move ###
        ### for example for for a game ended after 5 moves:                ###
        ### overal_weights = array([ 0.2, -0.4,  0.6, -0.8,  1. ])         ###
        
        for state, move, weight in zip(states, moves, weights):
            yield state, move, weight
            
def BatchGenerator(n, state_gen, batch_size):
    """Generator of game states and corrensponding moves and weights grouped in batches."""
    states = np.zeros((batch_size, n**3), dtype=np.float32)
    moves = np.zeros((batch_size), dtype=np.int32)
    weights = np.zeros((batch_size), dtype=np.float32)
    
    while True:
        for i in range(batch_size):
            states[i], moves[i], weights[i] = next(state_gen)
        yield states, moves, weights
    
    