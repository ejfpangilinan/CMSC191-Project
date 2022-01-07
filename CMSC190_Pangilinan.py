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
from sklearn.model_selection import train_test_split


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
	def __init__(self, X, Y, mode):
		#initialized important variables

		self.inputs = X

		if mode==0:

			self.hidden_weights = np.random.uniform(low=-1.0, high=1.0, size=(self.inputs.shape[1],4))#weights of hidden layer
			self.hidden_bias = np.random.uniform(low=-1.0, high=1.0, size=(4,)) #bias of hidden layer

			self.output_weights = np.random.uniform(low=-1.0, high=1.0, size=(4,1)) #weights of output layer
			self.ouput_bias = np.random.uniform(low=-1.0, high=1.0, size=(1,))  #bias of the output layer


		elif mode == 1:

			# inp =  str(input("Enter model file name:"))
			# model_file = np.load(str(inp+'.npy'),allow_pickle='TRUE').item()
			model_file = np.load('model.npy',allow_pickle='TRUE').item()
			self.load_model(model_file)

		else:
			print("INVALID MODE")
			exit()
		

		self.a_out = Y.reshape(len(Y),1)#actual outputs
		self.p_outputs = np.zeros(Y.shape) #holds predicted outputs

		print(self.hidden_weights, self.hidden_bias)

		print(self.output_weights, self.ouput_bias)


	def load_model(self,model):

		self.hidden_weights = model['hidden_weights']
		self.hidden_bias = model['hidden_bias']

		self.output_weights = model['output_weights']
		self.ouput_bias = model['ouput_bias']
		
		print("MODEL LOADED SUCCESSFULLY")



	def save_model(self):

		model = {
			"hidden_weights":self.hidden_weights,
			"hidden_bias": self.hidden_bias,
			"output_weights": self.output_weights,
			"ouput_bias":self.ouput_bias,
		}

		#save model as npy file
		np.save('model.npy',model)
		print("MODEL SAVED SUCCESSFULLY AS model.npy")



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
	# Prepare the Datasets

	#CLEAN THE DATASET
	# load data
	test1 = read_csv('datatest.txt', header=0, index_col=0, parse_dates=True, squeeze=True)
	train = read_csv('datatraining.txt', header=0, index_col=0, parse_dates=True, squeeze=True)
	test2 = read_csv('datatest2.txt', header=0, index_col=0, parse_dates=True, squeeze=True)

	# vertically stack and maintain temporal order
	data = concat([test1, train, test2])

	values = data.values
	#separate inputs and output values
	INPUT, OUTPUT = values[:, 1:-1], values[:, -1]
	X, Y = np.array((INPUT),dtype=float), np.array((OUTPUT),dtype=float)



	# split the dataset
	trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, shuffle=False, random_state=1)

	#print dataset information
	# print(data.info())
	print("number of training set \n input: ", len(trainX),"\n output: ",len(trainY))
	print("number of test set \n input: ", len(testX),"\n output: ",len(testY))


	#get the frequency count of occupied(1) vs not occuppied(0) in the dataset
	unique, counts = np.unique(trainY, return_counts=True)
	freq_array = np.asarray((unique, counts),dtype=int).T

	#upsample the training Data for occuppied(1):
	up_x = []
	up_y = []

	r = (freq_array[0][1]-freq_array[1][1])
	add = 0
	cnt=0

	while add!=r:
	    if trainY[cnt]==1:
	        up_x.append(trainX[cnt])
	        up_y.append(trainY[cnt])
	        add+=1
	        
	    cnt+=1
	    if cnt>=len(trainY) and add<r:
	        cnt=0
	    
	        
	up_x = np.array((up_x),dtype=float)
	up_y = np.array((up_y),dtype=float)

	trainX = np.concatenate((trainX,up_x),axis=0)
	trainY = np.concatenate((trainY,up_y),axis=0)


	"""
	Independent Variables:
	-Temperature, Humidity, Light, CO2, HumidityRatio
	Dependent Variable: 
	-Occupancy


	"""
	#get the frequency count of occupied(1) vs not occuppied(0) in the dataset after upsampling
	unique, counts = np.unique(trainY, return_counts=True)
	freq_array = np.asarray((unique, counts),dtype=int).T



	NN = NeuralNetwork(trainX, trainY,0)
	training_loss = []
	test_loss = []

	for epoch in range(1500):

		if epoch % 100==0:
			print ("Epoch " + str(epoch) + " Loss: " + str(np.mean(np.square(NN.a_out- NN.p_outputs))), end=' ') # mean squared error for loss
			evaluation(NN.a_out,normalize(NN.p_outputs))
			training_loss.append(np.mean(np.square(NN.a_out- NN.p_outputs)))

		NN.train()

	print("EVALUATION TRAIN")
	evaluation(NN.a_out,normalize(NN.p_outputs))

	NN.save_model()

	NN = NeuralNetwork(testX, testY,1)

	for epoch in range(1500):

		if epoch % 100==0:
			print ("Epoch " + str(epoch) + " Loss: " + str(np.mean(np.square(NN.a_out- NN.p_outputs))), end=' ') # mean squared error for loss
			evaluation(NN.a_out,normalize(NN.p_outputs))
			test_loss.append(np.mean(np.square(NN.a_out- NN.p_outputs)))

		NN.train()

	x_axis = np.arange(0,1500, 100)

	# plotting
	plt.title("LOSS using Mean Squared Error")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.plot(x_axis,np.asarray(training_loss), color ="red", label ='Training')
	plt.plot(x_axis,np.asarray(test_loss), color ="blue", label ='Test')
	plt.legend()
	plt.show()


	
	print("EVALUATION TEST1")
	evaluation(testY,NN.predict(testX))


	


	


	