import time
import random
import gym
import numpy as np
import math
from gym import spaces

class Data(gym.Env):
    #metadata = {'render.modes': ['human']}

    def __init__(self, values=[5, 3, 1, 2, 4]):
        super(Data, self).__init__()
        self.SetValues(values)
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=5, shape=(1,5), dtype=np.int32)
        self._steps = 0
        self._isTraining = True

    def step(self, action):
        if self._isTraining:
            return self.TrainingStep(action)
        else:
            return self.TestingStep(action)

    def TrainingStep(self, action):        
        x, y = self.Interpret(action)
        state = self.Swap(x, y)
        reward, done = self.Reward()

        return state, reward, done, {}
        
    def TestingStep(self, action):
        self._steps += 1
        x, y = self.Interpret(action)
        state = self.Swap(x, y)
        done = False
        if self._steps == 4:
            if self.IsCorrect() == False:
                print('')
                print("WRONG")
                print('')
            done = True
        
        reward = 0
        return state, reward, done, {}

    def reset(self):
        self._steps = 0
        
        while self.IsCorrect():
            self.Shuffle()
            
        print("New state", self._values)
            
        return self._state

    def render(self, mode='human', close=False):
        print("step:", self._steps, self._values)
        
    def Shuffle(self):
        for i in range(50):
            x, y = self.Interpret(random.randint(0, 9))
            self.Swap(x, y)
            
    # Used for changing the values of the env
    def SetValues(self, values):
        self._values = values
        self._state = self.CompressValues(self._values)
        self._answer = sorted(self._values)
        
        print(self._values)
        print(self._state)
        print(self._answer)
        
    def CompressValues(self, values):
        larg = max(values)
        arr = []
        for v in values:
            arr.append(math.tanh(v/larg))
        return arr
        
    def Sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # TODO make dynamic
    def Interpret(self, action):
        x = None
        y = None
        if action == 0:
            x = 0
            y = 1
        elif action == 1:
            x = 0
            y = 2
        elif action == 2:
            x = 0
            y = 3
        elif action == 3:
            x = 0
            y = 4
        elif action == 4:
            x = 1
            y = 2
        elif action == 5:
            x = 1
            y = 3
        elif action == 6:
            x = 1
            y = 4
        elif action == 7:
            x = 2
            y = 3
        elif action == 8:
            x = 2
            y = 4
        elif action == 9:
            x = 3
            y = 4
        elif action == 10:
            x = 0
            y = 0

        return x, y

    # move into different object
    def Swap(self, x, y):
        tmp = self._state[x]
        self._state[x] = self._state[y]
        self._state[y] = tmp
        
        tmp = self._values[x]
        self._values[x] = self._values[y]
        self._values[y] = tmp

        # print("Current Data", self._state)
        return self._state

    def Reward(self):
        self._steps += 1
        # reward = -(2 + 0.01 * self._steps)
        reward = -2.5

        correct = self.IsCorrect()
        if correct:
            # reward = max(0,10/self._steps)
            reward = max(0,5/self._steps)
        return reward, correct
    
    def IsCorrect(self): 
        i = 0
        correct_count = 0
        for v in self._values:
            if v == self._answer[i]:
                correct_count += 1
            i += 1
                
        if correct_count == len(self._values):
            return True
            
        return False
        