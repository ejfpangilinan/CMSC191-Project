import numpy as np
import math
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import concat
import random


#define sigmoid activation function (0,1)
def sigmoid(x):
    s = 1 / (1 + math.exp(-x))
    return s

#CLEAN THE DATASET
# load data
test1 = read_csv('datatest.txt', header=0, index_col=0, parse_dates=True, squeeze=True)
train = read_csv('datatraining.txt', header=0, index_col=0, parse_dates=True, squeeze=True)
test2 = read_csv('datatest2.txt', header=0, index_col=0, parse_dates=True, squeeze=True)


values = train.values
#separate inputs and output values
INPUT, OUTPUT = values[:, 1:-1], values[:, -1]


"""
Independent Variables:
-Temperature, Humidity, Light, CO2, HumidityRatio

Dependent Variable: 
-Occupancy

#make a 3-layer (5-5-1) Neural Network

"""

#sample run .. get the value of hidden layer on first iteration

training_input = np.array(INPUT[0:10])
training_output =  np.array(OUTPUT[0:10]).T

hidden_layer = np.array([0,0,0,0,0]).T #bias is initially 0

#specialized init weigh representing combination(5,4)
weights = [[0.5,0.5,0.5,0.5,0],
			[0.5,0.5,0.5,0,0.5],
			[0.5,0.5,0,0.5,0.5],
			[0.5,0,0.5,0.5,0.5],
			[0,0.5,0.5,0.5,0.5]]



for i in range(1):

	#input- hidden layer

	for x in range(len(hidden_layer)):
		nw = np.array(weights[x]).T
		bias = random.randrange(-5,5)
		print(training_input[i],nw)
		print(bias)
		hidden_layer[x] = sigmoid(np.dot(training_input[i],nw)+bias)

	print("HIDDEN LAYER: ", hidden_layer)

	#hidden-output
	obias = random.randrange(-5,5)
	ow = np.array(hidden_layer).T
	output = sigmoid(np.dot(hidden_layer,ow)+obias)

	print("OUTPUT: ", output)




