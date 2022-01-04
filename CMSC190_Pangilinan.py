import numpy as np
import math
import pandas
from pandas import read_csv
from pandas import concat
import random
from decimal import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,recall_score,precision_score
from sklearn.metrics import f1_score,confusion_matrix,roc_auc_score


import warnings

#suppress warnings
warnings.filterwarnings('ignore')





#define sigmoid activation function
def sigmoid(x):
	return 1 / (1.0 + np.exp(-x))

# neuron derivatives
def n_derivatives(z):
	return z * (1.0 - z)


def confusion_matrix(pred, out):
	conf = {"tp": 0,
			"tn": 0,
			"fp": 0,
			"fn":0}

	for i in range(len(pred)):
		if pred[i]>=0.5 and out[i] == 1.0: #tp
			conf["tp"]+=1
		elif pred[i]<0.5 and out[i] == 0.0: #tn
			conf["tn"]+=1
		elif pred[i]>=0.5 and out[i] == 0.0: #fp
			conf["fp"]+=1
		elif pred[i]<0.5 and out[i] == 1.0: #fn
			conf["fn"]+=1
		else:
			pass

	return conf

def normalize(pred):
	ret = np.array((pred))

	for i in range(len(pred)):
		if pred[i]>=0.5: #tp
			ret[i]=1
		else:
			ret[i]=0

	return ret

def conf_accuracy(pred,out):
	confu = confusion_matrix(pred,out)

	return (confu["tp"]+confu["tn"]) / len(pred)


def evaluation(y_test,y_pred):

	acc = accuracy_score(y_test,y_pred)
	rcl = recall_score(y_test,y_pred)
	f1 = f1_score(y_test,y_pred)
	auc_score = roc_auc_score(y_test,y_pred)
	prec_score = precision_score(y_test,y_pred)

	metric_dict={'accuracy': round(acc,3),
	           'recall': round(rcl,3),
	           'F1 score': round(f1,3),
	           'auc score': round(auc_score,3),
	           'precision': round(prec_score,3) 
	          }

	return print(metric_dict)

class NeuralNetwork(object):
	"""Make a 3-layer network (5-4-1)"""
	def __init__(self, X, Y):
		#initialized important variables

		self.inputs = X

		self.hidden_weights = np.random.uniform(low=-0.3, high=0.3, size=(self.inputs.shape[1],5))#weights of hidden layer
		self.hidden_bias = np.random.uniform(low=-0.3, high=0.3, size=(5,)) #bias of hidden layer

		self.output_weights = np.random.uniform(low=-0.3, high=0.3, size=(5,1)) #weights of output layer
		self.ouput_bias = np.random.uniform(low=-0.3, high=0.3, size=(1,))  #bias of the output layer

		#------------------------------------------------------------------------------------------------------------#
		# self.hidden_weights = np.full((self.inputs.shape[1],3), 0.5, dtype=float) #weights of hidden layer
		# self.hidden_bias = np.full((3,), 0.5, dtype=float) #bias of hidden layer

		# self.output_weights = np.full((3,1), 0.5, dtype=float) #weights of output layer
		# self.ouput_bias = np.full((1,), 0.5, dtype=float) #bias of the output layer
		#------------------------------------------------------------------------------------------------------------#


		self.a_out = Y.reshape(len(Y),1)#actual outputs
		self.p_outputs = np.zeros(Y.shape) #holds predicted outputs

		print(self.hidden_weights, self.hidden_bias)

		print(self.output_weights, self.ouput_bias)

	def predict(self,X):
		#predict using a different input
		self.inputs = X
		return normalize(self.forward_pass())

	def forward_pass(self):
		self.hidden_nodes = sigmoid(np.add(np.dot(self.inputs, self.hidden_weights),self.hidden_bias)) #contains hidden node inputs
		outputs = sigmoid(np.add(np.dot(self.hidden_nodes, self.output_weights),self.ouput_bias)) #contain output prediction
		return outputs

	def back_propagration(self,l_rate):
		#output error and delta
		self.output_error = self.a_out - self.p_outputs
		self.output_delta = self.output_error * n_derivatives(self.p_outputs)

		#hidden error and delta
		self.hidden_error = np.dot(self.output_delta,self.output_weights.T)
		self.hidden_delta = self.hidden_error*n_derivatives(self.hidden_nodes)

		#update weights
		self.output_weights += np.dot(self.hidden_nodes.T,self.output_delta) * l_rate
		self.hidden_weights += np.dot(self.inputs.T,self.hidden_delta) * l_rate

		# #update biases
		self.hidden_bias  += np.sum(self.hidden_delta,axis=0) * l_rate
		self.ouput_bias += np.sum(self.output_delta,axis=0)  *l_rate
		return

	def train(self):
		self.p_outputs = self.forward_pass()
		self.back_propagration(0.00001) #use learning rate
		return
		

if __name__ == "__main__":
	#CLEAN THE DATASET
	# load data
	test1 = read_csv('datatest.txt', header=0, index_col=0, parse_dates=True, squeeze=True)
	train = read_csv('datatraining.txt', header=0, index_col=0, parse_dates=True, squeeze=True)
	test2 = read_csv('datatest2.txt', header=0, index_col=0, parse_dates=True, squeeze=True)

	values = train.values
	#separate inputs and output values
	INPUT, OUTPUT = values[:, 1:-1], values[:, -1]
	X, Y = np.array((INPUT),dtype=float), np.array((OUTPUT),dtype=float)

	values1 = test1.values
	#separate inputs and output values
	INPUT1, OUTPUT1 = values1[:, 1:-1], values1[:, -1]
	X1, Y1 = np.array((INPUT1),dtype=float), np.array((OUTPUT1),dtype=float)

	values2 = test2.values
	#separate inputs and output values
	INPUT2, OUTPUT2 = values2[:, 1:-1], values2[:, -1]
	X2, Y2 = np.array((INPUT2),dtype=float), np.array((OUTPUT2),dtype=float)

	print(train.info())


	"""
	Independent Variables:
	-Temperature, Humidity, Light, CO2, HumidityRatio
	Dependent Variable: 
	-Occupancy
	"""


	NN = NeuralNetwork(X, Y)

	for epoch in range(1500):	
		if epoch % 100==0: 
			print ("Epoch " + str(epoch) + " Loss: " + str(np.mean(np.square(NN.a_out- NN.p_outputs)))) # mean squared error for loss
		NN.train()

	# print("TRAINING")
	# print("Data points:",len(X))
	# print ("loss1:" + str(np.mean(np.square(NN.a_out- NN.p_outputs))))
	# print("Confusion Matrix:",confusion_matrix(NN.p_outputs,NN.a_out))
	# print("Confusion Accuracy:",conf_accuracy(NN.p_outputs,NN.a_out))
	

	
	# print("TEST1")
	# print("Data points:",len(X1))
	# print ("loss1:" + str(np.mean(np.square(Y1- NN.predict(X1)))))
	# print("Confusion Matrix:",confusion_matrix(NN.predict(X1),Y1))
	# print("Confusion Accuracy:",conf_accuracy(NN.predict(X1),Y1))


	# print("TEST2")
	# print("Data points:",len(X2))
	# print ("loss2:" + str(np.mean(np.square(Y2- NN.predict(X2)))))
	# print("Confusion Matrix:",confusion_matrix(NN.predict(X2),Y2))
	# print("Confusion Accuracy:",conf_accuracy(NN.predict(X2),Y2))



	print("EVALUATION TRAIN")
	evaluation(NN.a_out,normalize(NN.p_outputs))
	print("EVALUATION TEST1")
	evaluation(Y1,NN.predict(X1))
	print("EVALUATION TEST2")
	evaluation(Y2,NN.predict(X2))