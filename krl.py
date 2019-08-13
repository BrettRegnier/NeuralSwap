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

