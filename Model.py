import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
import numpy as np
import random

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
        
        self._inputCount = 5 # TODO change to allow for dynamic sizing.
        self._outputCount = 11 # TODO change to function for calculating choose.
        
        # self._session = tf.Session() # replaced by keras
        self.DefineModel()
        # self._session.run(self._initializer) # replaced by keras
        
    def DefineModel(self):
        self._classifier = Sequential()
        self._classifier.add(Dense(units=64, activation='sigmoid', input_dim=self._inputCount))
        self._classifier.add(Dense(units=64, activation='sigmoid'))
        self._classifier.add(Dense(units=self._outputCount, activation='softmax'))
        self._classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
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
        
        self._classifier.fit(state, state_Q_values)
        
    def Update(self, state, next_state, action, reward):
        self.Train([[state]], action, reward, [[next_state]])
        
        if self._ep > 0:
            self._ep -= self._epdelt