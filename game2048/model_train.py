import numpy as np
from game import Game
from expectimax import board_to_move
import random

class ModelTrain:
    def __init__(self,model):
        self.OUT_SHAPE = (4,4)
        self.CAND = 12
        self.map_table = {2**i: i for i in range(1,12)}
        self.map_table[0]=0
        self.model = model
        
    def grid_ohe(self,arr):
        ret = np.zeros(shape=self.OUT_SHAPE + (self.CAND,), dtype=bool)
        for r in range(self.OUT_SHAPE[0]):
            for c in range(self.OUT_SHAPE[1]):
                ret[r,c,self.map_table[arr[r,c]]] = 1
        return ret
    
    def dtoy(self,x):
        ret = np.zeros((1,4))
        ret[0,x]=1
        return ret
    
    def ytod(self,x):
        return x.argmax(axis=1)[0]
    
    def data(self,data_num,end_score):
        x_train = np.zeros((0,4,4,12))
        y_train = np.zeros((0,4))
        num = 0
        
        while num < data_num:
            game = Game(4,end_score)
            while game.end == 0 and num < data_num:
                gb = game.board
                x = self.grid_ohe(gb)
                x = np.array([x])
                x_train = np.vstack((x_train,x))
                y_t = np.array([gb])
                d = board_to_move(y_t[0])
                y_train = np.vstack((y_train,self.dtoy(d)))
                y = self.model.predict(x)
                game.move(self.ytod(y))
                num += 1
        
        return x_train,y_train
    
    def train(self,x,y,train_size,train_num):
        for j in range(train_num):
            train_sequence = random.sample(range(len(y)),train_size)
            x_train = np.zeros((0,4,4,12))
            y_train = np.zeros((0,4))
            for i in train_sequence:
                x_train = np.vstack((x_train,np.array([x[i]])))
                y_train = np.vstack((y_train,np.array([y[i]])))
            self.model.fit(x_train,y_train,batch_size = len(x_train))
            self.model.save('model.h5')
    
    def test(self,n_test):
        score = 0
        l = []
        for i in range(n_test):
            game = Game(4,2048)
            while game.end == 0:
                gb = game.board
                x = self.grid_ohe(gb)
                x = np.array([x])
                y = self.model.predict(x)
                game.move(self.ytod(y))
            score = score + game.score
            l.append(game.score)
        aver_score = int(score / n_test)
        print(aver_score)
        print(l)

