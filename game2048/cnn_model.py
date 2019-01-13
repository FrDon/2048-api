# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 12:54:25 2019

@author: syd
"""
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import Dense,Activation,Conv2D,Flatten,Input,BatchNormalization
from keras.optimizers import Adam

inputs = Input((4,4,12))

conv = inputs
FILTERS = 128
conv41 = Conv2D(filters=FILTERS,kernel_size=(4,1),kernel_initializer='he_uniform')(conv)
conv14 = Conv2D(filters=FILTERS,kernel_size=(1,4),kernel_initializer='he_uniform')(conv)
conv22 = Conv2D(filters=FILTERS,kernel_size=(2,2),kernel_initializer='he_uniform')(conv)
conv33 = Conv2D(filters=FILTERS,kernel_size=(3,3),kernel_initializer='he_uniform')(conv)
conv44 = Conv2D(filters=FILTERS,kernel_size=(4,4),kernel_initializer='he_uniform')(conv)

hidden = concatenate([Flatten()(conv41),Flatten()(conv14),Flatten()(conv22),Flatten()(conv33),Flatten()(conv44)])
x = BatchNormalization()(hidden)
x = Activation('relu')(hidden)

for width in [512,128]:
    x = Dense(width,kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
outputs = Dense(4, activation='softmax')(x)
model = Model(inputs,outputs)
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.save('model.h5')
