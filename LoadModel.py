import pandas
from pandas import read_csv
from pandas import concat
import numpy as np
from sklearn.metrics import accuracy_score,recall_score,precision_score
from sklearn.metrics import f1_score,confusion_matrix,roc_auc_score
from sklearn.model_selection import train_test_split
from CMSC190_Pangilinan import *


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
print(data.info())
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


NN = NeuralNetwork(trainX, trainY,1)
training_loss = []
test_loss = []

for epoch in range(1500):

	if epoch % 100==0:
		print ("Epoch " + str(epoch) + " Loss: " + str(np.mean(np.square(NN.a_out- NN.p_outputs))), end=' ') # mean squared error for loss
		evaluation(NN.a_out,normalize(NN.p_outputs))
		training_loss.append(np.mean(np.square(NN.a_out- NN.p_outputs)))

	NN.train()

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


print("EVALUATION TRAIN")
evaluation(NN.a_out,normalize(NN.p_outputs))
print("EVALUATION TEST1")
evaluation(testY,NN.predict(testX))





print("TRAINING")
print("Data points:",len(trainX))
print ("loss1:" + str(np.mean(np.square(NN.a_out- NN.p_outputs))))
print("Confusion Matrix:",confusion_matrix(NN.p_outputs,NN.a_out))
print("Confusion Accuracy:",conf_accuracy(NN.p_outputs,NN.a_out))


#sample prediction

#23.7,26.272,585.2,749.2,0.00476416302416414,1
#x = ["Temperature","Humidity","Light","CO2","HumidityRatio"]
#y = ["Actual Occupancy"]

x= [23.7,26.272,585.2,749.2,0.00476416302416414]
y= [1]

print(normalize(NN.predict(x)),y)