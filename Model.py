import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.optimizers import Adam
import numpy as np
import random

import keras.backend as K
import time
from collections import deque

# lr = learning rate
# gamma = discount rate
# ep = exploration rate
# epdelt = eploration rate min

class Model:
    def __init__(self, lr=0.1, gamma=0.95, epsilon=1.0, iterations=10000):
        self._lr = lr
        self._gamma = gamma
        self._ep = epsilon
        self._epdelt = 1.0/iterations
        self._memory = deque(maxlen=2000)
        
        # different algorithm
        self._epmin = 0.01
        self._epdecay = 0.99
        
        self._inputCount = 5 # TODO change to allow for dynamic sizing.
        self._outputCount = 10 # TODO change to function for calculating choose.
        self._reward = 0
        
        # self._session = tf.Session() # replaced by keras
        self.DefineModel()
        # self._session.run(self._initializer) # replaced by keras
        
    def customLoss(self):
        def loss(y_true, y_pred):
            v = ((self._reward + self._gamma * y_pred) - y_true)
            return v * v
        
        return loss
    
    def HuberLoss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))
    
    def DefineModel(self):
        self._classifier = Sequential()
        self._classifier.add(Dense(units=128, activation='tanh', input_dim=self._inputCount))
        self._classifier.add(Dense(units=128, activation='tanh'))
        self._classifier.add(Dense(units=32, activation='tanh'))
        self._classifier.add(Dense(units=16, activation='sigmoid'))
        self._classifier.add(Dense(units=16, activation='sigmoid'))
        self._classifier.add(Dense(units=self._outputCount, activation='softmax'))
        self._classifier.compile(optimizer=Adam(lr=self._lr), loss=self.HuberLoss, metrics=['accuracy'])
        # self._classifier.compile(optimizer=Adam(lr=self._lr), loss="categorical_crossentropy", metrics=['accuracy'])
        
    def Act(self, state):
        actions = self._classifier.predict(state)		
        return actions
    
    def NextAction(self, state):
        # if the random is greater than ep then don't explore
        if random.random() > self._ep:
            return self.Greedy([[state]])
        else:
            return self.RandomAction()
    
    def Greedy(self, state):
        return np.argmax(self.Act(state))
        
    def RandomAction(self):
        # TODO make dynamic based on outputs
        return random.randint(0, self._outputCount-1)
        
    def Train(self, state, action, reward, next_state):
        state_Q_values = self.Act(state)
        next_Q_values = self.Act(next_state)
        
        print(state_Q_values)
        state_Q_values[0][action] = reward + self._gamma * np.amax(next_Q_values)
        
        self._reward = reward
        self._curstate = state_Q_values
        self._nextstate = next_Q_values
        
        self._classifier.fit(state, state_Q_values)
        
    def Update(self, state, next_state, action, reward):
        self.Train([[state]], action, reward, [[next_state]])
        
        if self._ep > 0:
            self._ep -= self._epdelt
            
    def AddMemory(self, state, action, next_state, reward, done):
        self._memory.append((state, action, reward, next_state, done))
        
    def Replay(self, batch_size):
        minibatch = random.sample(self._memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.Act([[state]])
            if done:
                target[0][action] = reward
            else:
                t = self.Act([[next_state]])[0]
                target[0][action] = reward + self._gamma * np.amax(t)
            self._classifier.fit([[state]], target, epochs=1, verbose=0)
        if self._ep > self._epmin:
            self._ep *= self._epdecay
            
    def ClearMemory(self):
        self._memory = deque(maxlen=2000)