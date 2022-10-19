#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:38:29 2022

@author: G.Gradziuk
"""

import numpy as np
import random
import ast
import time

def print_state(state, O_sym='O', X_sym='X', E_sym='‾'):
    if len(state.shape) == 1:
        n = int(np.cbrt(state.shape[-1]))
        assert n**3 == state.shape[-1]
        state = state.reshape((n, n, n))
    else:
        n = state.shape[-1]
        assert len(state.shape) == 3 and np.all(np.equal(state.shape, n))
   
    mark_dict = {1 : O_sym, -1 : X_sym, 0 : E_sym}
    state = state.reshape((n, n, n)).astype(np.int32)
    
    #print(" ".join([" " + n * "_ "] * n))
    for row in range(n):
        print(" ".join(["|" + "".join([mark_dict[c] + "|" for c in state[board, row, :]]) for board in range(n)]))
    print(" ".join([" " + n * "‾ "] * n))
    
class Game:
    def __init__(self, n, init_state=None, strategy=None, O_sym='O', X_sym='X'):
        if init_state is None:
            self.state = np.zeros(n ** 3, dtype=np.int32)
        else:
            assert n ** 3 == len(init_state)
            self.state = init_state
        
        self.state3d = self.state.reshape((n, n, n))
        self.empty_fields = set(range(n ** 3))
        self.turn = 1
        self.n = n
        self.last_field_tuple = None
        self.last_field = None
        self.winner = 0
        self.strategy = strategy
        self.O_sym = O_sym
        self.X_sym = X_sym
        
    def reset(self):
        self.__init__(self.n)
    
    def check_win(self):
        x, y, z = self.last_field_tuple
        # wins along cartesian axes
        win = any([np.all(np.equal(self.state3d[:, y, z], self.turn)),
                   np.all(np.equal(self.state3d[x, :, z], self.turn)),
                   np.all(np.equal(self.state3d[x, y, :], self.turn))])
        if win: return True
        
        # wins along x=+-y, x=+-z, y=+-z
        ran = list(range(self.n))
        nar = ran[::-1]
        if y == z:
            win = all([self.state3d[x, _y, _z] == self.turn for _y, _z in zip(ran, ran)])
            if win: return True
        if y == -z:
            win = all([self.state3d[x, _y, _z] == self.turn for _y, _z in zip(ran, nar)])
            if win: return True
           
        if x == z:
            win = all([self.state3d[_x, y, _z] == self.turn for _x, _z in zip(ran, ran)])
            if win: return True
        if x == -z:
            win = all([self.state3d[_x, y, _z] == self.turn for _x, _z in zip(ran, nar)])
            if win: return True
             
        if x == y:
            win = all([self.state3d[_x, _y, z] == self.turn for _x, _y in zip(ran, ran)])
            if win: return True
        if x == -y:
            win = all([self.state3d[_x, _y, z] == self.turn for _x, _y in zip(ran, nar)])
            if win: return True
            
        # wins along long diags
        win = any([all([self.state3d[_x, _y, _z] == self.turn for _x, _y, _z in zip(ran, ran, ran)]),
                   all([self.state3d[_x, _y, _z] == self.turn for _x, _y, _z in zip(ran, ran, nar)]),
                   all([self.state3d[_x, _y, _z] == self.turn for _x, _y, _z in zip(ran, nar, ran)]),
                   all([self.state3d[_x, _y, _z] == self.turn for _x, _y, _z in zip(ran, nar, nar)])
                  ])
        return win
            
            
            
    def make_move(self, field=None):
        if isinstance(field, tuple):
            field_tuple = field
            field = int(field[0] * self.n**2 + field[1] * self.n + field[2])
        else:
            if field is None:
                if self.strategy == None:
                    #make random move #TODO: chosing random field is not efficient
                    field = random.choice(list(self.empty_fields))
                else:
                    # Strategy is the ML decision model.
                    # Multiply by self.turn, so that current player's fields
                    # in self.state are 1 and opponents fields are -1.
                    model_input = np.reshape(self.state * self.turn, (1,-1))
                    move_probs = self.strategy(model_input)[0]
                    candidates = np.argsort(move_probs)
                    for ind in reversed(candidates):
                        if ind in self.empty_fields:
                            field = ind
                            break
                    
            #assert(isinstance(field, int))
            field_tuple = (field // self.n**2, field % self.n**2 // self.n, field % self.n)
            
        if not field in self.empty_fields:
            print("This field is already taken! Choose a different one!")
            return False
        
        self.empty_fields.remove(field)
        self.last_field = field
        self.last_field_tuple = field_tuple
        self.state[field] = self.turn
        self.state3d[field_tuple] = self.turn
        if self.check_win():
            self.winner = self.turn
        self.turn *= -1
        return True
    
    def print_state(self):
        mark_dict = {1 : self.O_sym, -1 : self.X_sym, 0 : '‾'}
        for row in range(self.n):
            print(" ".join(["|" + "".join([mark_dict[c] + "|" for c in self.state3d[board, row, :]]) for board in range(self.n)]))
        print(" ".join([" " + self.n * "‾ "] * self.n))
        
    def play(self, turn=1):
        self.turn = turn
        if self.turn == 1:
            self.print_state()
        while(True):
            if self.winner != 0: #TODO: or no move moves possible
                print(self.O_sym if self.winner == 1 else self.X_sym, " wins!")
                return self.winner
            elif len(self.empty_fields) == 0:
                print("It's a draw!")
                return self.winner
            
            if self.turn == 1:
                print("Your turn!")
                end_turn = False
                while(not end_turn):
                    field = ast.literal_eval(input("Enter the coordinates: "))
                    end_turn = self.make_move(field)
            else:
                print("Computer's move!")
                time.sleep(2)
                self.make_move()
                
            self.print_state()
                    
                    
            
        
    
    
    
    
        
    
    
    
        
        
    
        