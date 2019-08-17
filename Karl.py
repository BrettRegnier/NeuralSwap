import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import matplotlib.pyplot as plt

from Env import Data

class Karl:
    def __init__(self, env, name='Karl.h5', load=True, memory=5000, training_steps=20000):
        self._env = env
        self._memory = memory
        self._ts = training_steps
        self.CreateModel()

        if load: self.Load(name)
        
    def CreateModel(self):        
        # Build Karl
        nb_actions = self._env.action_space.n
        memory = SequentialMemory(limit=self._memory, window_length=1)
        policy = BoltzmannQPolicy()
        self._model = Sequential()
        self._model.add(Flatten(input_shape=self._env.observation_space.shape))
        # self._model.add(Dense(units=nb_actions, activation='sigmoid', input_dim=nb_actions))
        self._model.add(Dense(units=256, activation='relu'))
        self._model.add(Dense(units=512, activation='relu'))
        self._model.add(Dense(units=256, activation='relu'))
        self._model.add(Dense(units=128, activation='relu'))
        self._model.add(Dense(units=64, activation='relu'))
        self._model.add(Dense(units=nb_actions, activation='linear'))
        self._dqn = DQNAgent(model=self._model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2,
                    policy=policy)
        self._dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        
    def Load(self, name):
        if os.path.isfile("./" + name):
            self._model.load_weights(name)
        else:
            print("No save found with the name", name)
    
    def Save(self, name):
        self._model.save_weights(name)
        
    def Train(self):        
        # Train Karl and save the plot points
        history = self._dqn.fit(self._env, nb_steps=20000, visualize=False, verbose=2)
        
        # Save Karl
        self.Save("Karl.h5")
        
        # Plot training points
        plt.plot(history.history['nb_episode_steps'], linewidth=1.0)
        plt.title('Karl Fitness')
        plt.ylabel('Number of steps per episode')
        plt.xlabel('Episode')
        plt.show()
        
    def Test(self, values=None):
        self._env._isTraining = False
        if values != None:
            self._env.SetValues(values)
        
        self._dqn.test(self._env, nb_episodes=100, visualize=True)
        