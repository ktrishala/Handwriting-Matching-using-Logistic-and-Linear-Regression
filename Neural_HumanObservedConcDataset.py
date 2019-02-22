
# coding: utf-8

# In[41]:


import numpy as np
import csv
import math
import pandas as pd
from keras.models import Sequential  #importing to build the sequential model
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras import regularizers
from keras import optimizers


# In[42]:


def GetTargetVector(filePath):    #reading the entire target values into an numpy vector
    data = pd.read_csv(filePath)
    t=[]
    t=data['target'].values  
    print(np.shape(t))
    return t

def GetRawData(filePath):      #reading the entire input features into numpy matrix
    data = pd.read_csv(filePath)
    t=[]
    t=data.as_matrix(columns=data.columns[1:])
    print(np.shape(t))
    return t

def GenerateTrainingTarget(RawTarget,TrainingPercent):
    TrainingLen = int(math.ceil(len(RawTarget)*(TrainingPercent*0.01))) #Taking only 80% of the Target data for training
    t           = RawTarget[:TrainingLen]
    return t

def GenerateTrainingDataMatrix(RawData, TrainingPercent):
    T_len = int(math.ceil(len(RawData)*0.01*TrainingPercent)) #Taking only 80% of the data matrix for training
    d2 = RawData[:T_len]
    return d2


def GenerateTestingTarget(RawTarget,TestingPercent):
    T_len = int(math.ceil(len(RawTarget)*(TestingPercent*0.01))) #Taking only 80% of the Target data for training
    t           = RawTarget[-T_len:]
    return t

def GenerateTestingDataMatrix(RawData, TestingPercent):
    T_len = int(math.ceil(len(RawData)*0.01*TestingPercent)) #Taking only 80% of the data matrix for training
    d2 = RawData[-T_len:]
    return d2


# In[43]:


RawTarget = GetTargetVector('HumanObservedDatasetwithFeatureConcatenation.csv')
RawData = GetRawData('HumanObservedDatasetwithFeatureConcatenation.csv')
TrainingTarget=np.array(GenerateTrainingTarget(RawTarget,80))
TrainingData=np.array(GenerateTrainingDataMatrix(RawData,80))
TestingTarget=np.array(GenerateTestingTarget(RawTarget,20))
TestingData=np.array(GenerateTestingDataMatrix(RawData,20))


# In[44]:


input_size = 18
#drop_out = 0.2
first_dense_layer_nodes  = 1024
second_dense_layer_nodes  = 512
third_dense_layer_nodes  = 256

last_dense_layer_nodes = 1

def get_model():
    
    model = Sequential()  
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size,kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))       
    model.add(Dropout(0.3)) #dropout helps in regularization by randomly dropping out few neurons during training
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(third_dense_layer_nodes))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))   
    
    model.add(Dense(last_dense_layer_nodes))
    model.add(Activation('sigmoid'))    #implementing the sigmoid function for the final output layer 
    # used for binary classification method as it calculates the probability of the target class 
    #over all possible target classes. The class with the highest probability is the correct target class
    
    model.summary()     #prints a summary representation of the model 
    
    # To quantify the difference between the two probability distribution , i.e the one hot distribution/true distribution and predicted distribution
    model.compile(optimizers.Adam(lr=0.0001),   #Adam is a type of Gradient descent optimization algorithms
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])  
    
    return model


# In[45]:


model = get_model()


# In[46]:


validation_data_split = 0.2  #20% of the training dataset to be used for validation
num_epochs = 10000           #no. of times all of the training data is used to train the weights
model_batch_size = 128       #Number of samples per gradient update
tb_batch_size = 32   #number of input data for histogram computation
early_patience = 100  #number of epochs with no improvement after which training will be stopped.

#visualization of dynamic graphs of training and test metrics, 
#as well as activation histograms for the different layers in model
tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
#To Stop training when a monitored quantity has stopped improving
#val_loss-monitor the test (validation) loss at each epoch
#in min mode, training will stop when the quantity monitored has stopped decreasing
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

# Process Dataset
processedData=TrainingData
processedLabel = TrainingTarget
history = model.fit(processedData       #Numpy array of training data
                    , processedLabel    #Numpy array of target label data
                    , validation_split=validation_data_split #fraction of data to be kept for validation
                    , epochs=num_epochs
                    , batch_size=model_batch_size            #Number of samples per gradient update
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )


# In[47]:


#%matplotlib inline  

#magic func the output of plotting commands is displayed inline within frontends like the Jupyter notebook
#df = pd.DataFrame(history.history)  #stores the output as a dataframe ??history
#df.plot(subplots=True, grid=True, figsize=(10,15)) #Make separate subplots for each column; 


# In[48]:


wrong   = 0
right   = 0

for i,j in zip(TestingData,TestingTarget):   #iterating over the test input features and actual labels
    y = model.predict(np.array(i).reshape(-1,18)) #Generates output predictions for the input samples; reshape converts the input into a column matrix
    t=np.around(y)
    
    if j == int(t[0][0]):
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str((right/(right+wrong))*100))

