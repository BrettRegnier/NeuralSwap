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

gamma = 0.95    # discount rate
e = 1.0  # exploration rate
e_min = 0.01
decay = 0.99
lr = 0.001

permutations = 10
data = [[[5, 4, 3, 2, 1]]]
data_next = data

y = [[[5, 4, 3, 2, 1]]]

global isCorrect 
isCorrect = False

def Main():

	# 5 inputs, 1-5
	classifier = Sequential()
	classifier.add(Dense(units=64, activation='sigmoid', input_dim=5))
	classifier.add(Dense(units=64, activation='sigmoid'))
	classifier.add(Dense(units=permutations, activation='softmax'))
	classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

	while not isCorrect:
		prediction, action = Act(classifier)
		Interpret(action)
		Reward(classifier, prediction, action)
		
	
def Act(classifier):
	# random exploration
	# if np.random.rand() <= e:
	# 	return random.randrange(permutations)
	# else:		
	prediction = classifier.predict(data)
	selection = np.argmax(prediction)
	print("Probabilities", prediction)
	print("Selection", selection, "| Value of:", prediction[0][selection])
	return prediction, selection
	
def Interpret(action):
	if action == 0:
		Swap(0, 1)
	elif action == 1:
		Swap(0, 2)
	elif action == 2:
		Swap(0, 3)
	elif action == 3:
		Swap(0, 4)
	elif action == 4:
		Swap(1, 2)
	elif action == 5:
		Swap(1, 3)
	elif action == 6:
		Swap(1, 4)
	elif action == 7:
		Swap(2, 3)
	elif action == 8:
		Swap(2, 4)
	elif action == 9:
		Swap(3, 4)
	
def Reward(classifier, prediction, action):
	global isCorrect
	
	tr = 0 # total reward
	if data[0][0][0] == y[0][0]:
		tr += 1
	if data[0][0][1] == y[0][0][1]:
		tr += 1
	if data[0][0][1] == y[0][0][2]:
		tr += 1
	if data[0][0][1] == y[0][0][3]:
		tr += 1
	if data[0][0][1] == y[0][0][4]:
		tr += 1
		
	if tr == 5:
		isCorrect = True
		
	prediction[0][action] += tr
		
	classifier.fit(data, prediction, epochs=1, verbose=0)

def Swap(x, y):
	tmp = data[0][0][x]
	data[0][0][x] = data[0][0][y]
	data[0][0][y] = tmp
	print(data)
	
if __name__ == "__main__":
	Main()