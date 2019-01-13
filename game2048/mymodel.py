# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 19:29:04 2019

@author: syd
"""

from keras.models import load_model
import numpy as np

OUT_SHAPE = (4,4)
CAND = 12
map_table = {2**i: i for i in range(1,CAND)}
map_table[0]=0

def grid_ohe(arr):
    ret = np.zeros(shape=OUT_SHAPE + (CAND,), dtype=bool)
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            ret[r,c,map_table[arr[r,c]]] = 1
    return ret

def dtoy(x):
    ret = np.zeros((1,4))
    ret[0,x]=1
    return ret

def ytod(x):
    return x.argmax(axis=1)[0]

def oheboard(board):
    x = grid_ohe(board)
    return np.array([x])

       
class _model_:
    def __init__(self):
        self.model=load_model('./game2048/model.h5')
    def predict(self,board):
        Real_board = oheboard(board)
        direction = self.model.predict(Real_board)
        move = direction.argmax()
        return move