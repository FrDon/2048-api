import numpy as np


class Agent:
    '''Agent Base.'''
    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                #dataframe = pd.DataFrame({'move':direction},index = [0])
                #dataframe.to_csv("train.csv",index = False,sep=',')
                if self.display is not None:
                    self.display.display(self.game)
                    #print(self.game)
                    #dataframe = pd.DataFrame({'a_name':self.game})
                    #dataframe.to_csv("train.csv",index = False,sep=',')
        
                 
    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction

 
class MyAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .mymodel import _model_
        self.search_func_ = _model_()
  
    def step(self):
        direction = self.search_func_.predict(self.game.board)
        return direction
