import time
import random

class Data:
    def __init__(self, values=[5, 3, 1, 2, 4]):
        self._values = values
        self._state = self._values
        
        # TODO add cruncher
        # TODO add sorting
        self._answer = [1, 2, 3, 4, 5]
        
    def Shuffle(self):
        self._state = self._values
        for i in range(50):
            x, y = self.Interpret(random.randint(0, 9))
            self.Swap(x, y)
    
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
        
    def Act(self, action):
        x, y = self.Interpret(action)
        state = self.Swap(x, y)
        reward, done = self.Reward()
        
        return state, reward, done
        
    def Reward(self):
        i = 0
        reward = 0
        correct_count = 0
        done = False
        for v in self._state:
            if v == self._answer[i]:
                correct_count += 1
                
            if correct_count == len(self._state):
                reward = 1
                done = True
                
            i += 1
            
        return reward, done
            