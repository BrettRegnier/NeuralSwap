import time
import random
import gym
from gym import spaces
import numpy as np

class Data(gym.Env):
    #metadata = {'render.modes': ['human']}

    def __init__(self, values=[5, 3, 1, 2, 4]):
        super(Data, self).__init__()
        self._values = values
        self._state = self._values
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=5, shape=(1,5), dtype=np.int32)
        self._steps = 0

        # TODO add cruncher
        # TODO add sorting
        self._answer = [1, 2, 3, 4, 5]

    def step(self, action):
        x, y = self.Interpret(action)
        state = self.Swap(x, y)
        reward, done = self.Reward()

        return state, reward, done, {}

    def reset(self):
        self._state = self._values
        self._steps = 0
        for i in range(50):
            x, y = self.Interpret(random.randint(0, 9))
            self.Swap(x, y)
        return self._state

    def render(self, mode='human', close=False):
        print(self._state)

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

        # print("Current Data", self._state)
        return self._state

    def Reward(self):
        self._steps += 1
        i = 0
        reward = -1
        correct_count = 0
        done = False
        for v in self._state:
            if v == self._answer[i]:
                correct_count += 1

            if correct_count == len(self._state):
                reward = max(0,5-self._steps)
                done = True

            i += 1

        return reward, done
