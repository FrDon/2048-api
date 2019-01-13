import numpy as np
from keras.models import load_model
from game import Game
from expectimax import board_to_move

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

def model_train(model,aim,n_test,end_score):        
    score = 0
    train_number = 0
    list = []
    while score < aim :
        score = 0
        train_number = train_number + 1
        if train_number % 21 == 0 :
            list = []
        x_train64 = np.zeros((0,4,4,12))
        y_train64 = np.zeros((0,4))
        x_train128 = np.zeros((0,4,4,12))
        y_train128 = np.zeros((0,4))
        x_train256 = np.zeros((0,4,4,12))
        y_train256 = np.zeros((0,4))
        x_train512 = np.zeros((0,4,4,12))
        y_train512 = np.zeros((0,4))
        x_train1024 = np.zeros((0,4,4,12))
        y_train1024 = np.zeros((0,4))
        x_train2048 = np.zeros((0,4,4,12))
        y_train2048 = np.zeros((0,4))        
                       
        for i in range(n_test):
            game = Game(4,end_score)
            x_t64 = np.zeros((0,4,4,12))
            y_t64 = np.zeros((0,4,4))
            x_t128 = np.zeros((0,4,4,12))
            y_t128 = np.zeros((0,4,4))
            x_t256 = np.zeros((0,4,4,12))
            y_t256 = np.zeros((0,4,4))
            x_t512 = np.zeros((0,4,4,12))
            y_t512 = np.zeros((0,4,4))
            x_t1024 = np.zeros((0,4,4,12))
            y_t1024 = np.zeros((0,4,4))
            x_t2048 = np.zeros((0,4,4,12))
            y_t2048 = np.zeros((0,4,4))
    
            while game.end == 0 and game.score < 64 :
                gb = game.board
                x = grid_ohe(gb)
                x = np.array([x])
                x_t64 = np.vstack((x_t64,x))
                y_t64 = np.vstack((y_t64,np.array([gb])))                                             
                y = model.predict(x)
                game.move(ytod(y))
            while game.end == 0 and game.score < 128 :
                gb = game.board
                x = grid_ohe(gb)
                x = np.array([x])
                x_t128 = np.vstack((x_t128,x))
                y_t128 = np.vstack((y_t128,np.array([gb])))                                             
                y = model.predict(x)
                game.move(ytod(y))
            while game.end == 0 and game.score < 256 :
                gb = game.board
                x = grid_ohe(gb)
                x = np.array([x])
                x_t256 = np.vstack((x_t256,x))
                y_t256 = np.vstack((y_t256,np.array([gb])))                                
                y = model.predict(x)
                game.move(ytod(y))
            while game.end == 0 and game.score < 512 :
                gb = game.board
                x = grid_ohe(gb)
                x = np.array([x])
                x_t512 = np.vstack((x_t512,x))
                y_t512 = np.vstack((y_t512,np.array([gb])))                                
                y = model.predict(x)
                game.move(ytod(y))
            while game.end == 0 and game.score < 1024 :
                gb = game.board
                x = grid_ohe(gb)
                x = np.array([x])
                x_t1024 = np.vstack((x_t1024,x))
                y_t1024 = np.vstack((y_t1024,np.array([gb])))                                
                y = model.predict(x)
                game.move(ytod(y))
            while game.end == 0 :
                gb = game.board
                x = grid_ohe(gb)
                x = np.array([x])
                x_t2048 = np.vstack((x_t2048,x))
                y_t2048 = np.vstack((y_t2048,np.array([gb])))                                
                y = model.predict(x)
                game.move(ytod(y))
           
            score = score + game.score
            print(game.score)
            
            if game.score < 2048 :
                if len(x_t64) > 0 :                    
                    x_train64 = np.vstack((x_train64,x_t64))
                    for j in range(len(y_t64)):
                        d = board_to_move(y_t64[j])
                        y_train64 = np.vstack((y_train64,dtoy(d)))
                    
                if len(x_t128) > 0 :                    
                    x_train128 = np.vstack((x_train128,x_t128))
                    for j in range(len(y_t128)):
                        d = board_to_move(y_t128[j])
                        y_train128 = np.vstack((y_train128,dtoy(d)))
                    
                if len(x_t256) > 0 :                    
                    x_train256 = np.vstack((x_train256,x_t256))
                    for j in range(len(y_t256)):
                        d = board_to_move(y_t256[j])
                        y_train256 = np.vstack((y_train256,dtoy(d)))
                        
                if len(x_t512) > 0 :                    
                    x_train512 = np.vstack((x_train512,x_t512))
                    for j in range(len(y_t512)):
                        d = board_to_move(y_t512[j])
                        y_train512 = np.vstack((y_train512,dtoy(d)))
                    
                if len(x_t1024) > 0  :                    
                    x_train1024 = np.vstack((x_train1024,x_t1024))
                    for j in range(len(y_t1024)):
                        d = board_to_move(y_t1024[j])
                        y_train1024 = np.vstack((y_train1024,dtoy(d)))
                        
                if len(x_t2048) > 0 :                    
                    x_train2048 = np.vstack((x_train2048,x_t2048))
                    for j in range(len(y_t2048)):
                        d = board_to_move(y_t2048[j])
                        y_train2048 = np.vstack((y_train2048,dtoy(d)))
                    
        if len(x_train64) > 0 :
            model.fit(x_train64,y_train64,epochs=1,batch_size= len(x_train64))
            
        if len(x_train128) > 0 :
            model.fit(x_train128,y_train128,epochs=1,batch_size= len(x_train128))
        
        if len(x_train256) > 0 :
            model.fit(x_train256,y_train256,epochs=1,batch_size= len(x_train256))
        
        if len(x_train512) > 0 :
            model.fit(x_train512,y_train512,epochs=1,batch_size= len(x_train512))
        
        if len(x_train1024) > 0 :
            model.fit(x_train1024,y_train1024,epochs=1,batch_size= len(x_train1024))
        
        if len(x_train2048) > 0 :
            model.fit(x_train2048,y_train2048,epochs=1,batch_size= len(x_train2048))
        
            
        model.save('model.h5')
    
        print()
        score  = score / n_test
        print(score)
        list.append(score)
        print(score)
        print(list)
        print(train_number)
        print()

        
model = load_model('model.h5') 
model_train(model,800,50,2048)

