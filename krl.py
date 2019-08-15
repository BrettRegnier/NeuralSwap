import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from Env import Data

import matplotlib.pyplot as plt

load = False

def Load(model, name):
    model.load_weights(name)

def Save(model, name):
    model.save_weights(name)

class Karl:
	def __init__(self, env, memory=5000, training_steps=20000):
		self._env
		self._memory = memory
		self._ts = training_steps
		self._model = self.CreateModel()
		
	def CreateModel(self):
		self._model = Sequential()
		self._model.add(Flatten(input_shape=env.observation_space.shape))
		self._model.add(Dense(units=nb_actions, activation='sigmoid', input_dim=nb_actions))
		self._model.add(Dense(units=256, activation='relu'))
		self._model.add(Dense(units=512, activation='relu'))
		self._model.add(Dense(units=256, activation='relu'))
		self._model.add(Dense(units=128, activation='relu'))
		self._model.add(Dense(units=64, activation='relu'))
		self._model.add(Dense(units=nb_actions, activation='linear'))
		
	def Load(self, name):
		self._model.load_weights(name)
	
	def Save(self, name):
		self._model.save_weights(name)
	
		
	def CreateModel(self):
		pass
	
env = Data()
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=env.observation_space.shape))
model.add(Dense(units=nb_actions, activation='sigmoid', input_dim=nb_actions))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=nb_actions, activation='linear'))

if load:
    Load(model, "Karl.h5")
else:
    # Train Karl
    memory = SequentialMemory(limit=5000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2,
                policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    history = dqn.fit(env, nb_steps=20000, visualize=False, verbose=2)
    Save(model, "Karl.h5")
plt.plot(history.history['nb_episode_steps'], linewidth=1.0)
#plt.plot(history.history['nb_steps'])
plt.title('Karl Fitness')
plt.ylabel('Number of steps per episode')
plt.xlabel('Episode')
#plt.legend(['Episode steps'], loc='upper left')
plt.show()

