import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from Env import Data

import matplotlib.pyplot as plt

env = Data()
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=env.observation_space.shape))
model.add(Dense(units=128, activation='relu', input_dim=nb_actions))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=nb_actions, activation='linear'))

memory = SequentialMemory(limit=100000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2,
               policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
history = dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
plt.plot(history.history['nb_episode_steps'])
#plt.plot(history.history['nb_steps'])
plt.title('Fuck')
plt.ylabel('Number of steps per episode')
plt.xlabel('Episode')
#plt.legend(['Episode steps'], loc='upper left')
plt.show()