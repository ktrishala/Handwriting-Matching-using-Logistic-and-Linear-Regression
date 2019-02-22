
# coding: utf-8

# In[81]:


from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv
import math


# In[82]:


TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 9


# In[83]:


def GetTargetVector(filePath):   #creating the target vector
    data = pd.read_csv(filePath)
    t=[]
    t=data['target'].values  
    print(np.shape(t))
    return t

def GetRawData(filePath):              #creating the input features matrix
    data = pd.read_csv(filePath)
    t=[]
    t=data.as_matrix(columns=data.columns[1:])
    dataMatrix = np.transpose(t) 
    return dataMatrix

def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01))) #Taking only 80% of the Target data for training
    t           = rawTraining[:TrainingLen]
    return t

def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent)) #Taking only 80% of the data matrix for training
    d2 = rawData[:,0:T_len]
    return d2

def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01)) #Taking only 10% of the data matrix for validation
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End] 
    return dataMatrix

def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01)) #Taking only 10% of the Target data for training
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    return t


def sigmoid(z):                #defining the sigmoid function
    return 1 / (1 + np.exp(-z))


def GetValTest(TEST_Data,W):     #generating the output labels for test data using the weights computed
    Y = np.dot(TEST_Data,W)
    return 1 / (1 + np.exp(-Y))

def GetErms(VAL_TEST_OUT,ValDataAct):  #calculating Erms and accuracy for actual and predicted values.
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)  #Sum of squared error
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]): #rounding off the output values and comparing with actual labels
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


# In[84]:


RawTarget = GetTargetVector('HumanObservedDatasetwithFeatureSubtraction.csv')
RawData = GetRawData('HumanObservedDatasetwithFeatureSubtraction.csv')


# In[85]:


TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))


# In[86]:


TrainingData=np.transpose(TrainingData)
print(TrainingData.shape)
ValData=np.transpose(ValData)
print(ValData.shape)
TestData=np.transpose(TestData)
print(TestData.shape)


# In[87]:


W = np.random.normal(0, 1, M).reshape(M)
W_Now=np.dot(abs(W),np.sqrt(1/(M+1))) #Initializing the weight matrix using Xavier’s random weight initialization
La           = 2
learningRate = 0.00003
L_Erms_Val   = []
L_Acc_Val    = []
L_Erms_TR    = []
L_Acc_TR     = []
L_Erms_Test  = []
W_Mat        = []
L_Acc_Test   = []


for i in range(0,1000):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    z = np.dot(TrainingData,W_Now)
    h = sigmoid(z)
    
    Delta_E_D     = np.dot(np.transpose(TrainingData), (h - TrainingTarget))
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W) 
    Delta_W       = -np.dot(learningRate,Delta_E) 
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next 
    
    
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TrainingData,W_T_Next)  #Generating the training labels based on updated weights
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget) #Getting RMS error and accuracy
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    L_Acc_TR.append(float(Erms_TR.split(',')[0]))

    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(ValData,W_T_Next)      #Generating the validation labels based on updated weights
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct) #Getting RMS error and accuracy
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    L_Acc_Val.append(float(Erms_Val.split(',')[0]))

    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TestData,W_T_Next)   #Generating the testing labels based on updated weights
    Erms_Test = GetErms(TEST_OUT,TestDataAct)     #Getting RMS error and accuracy
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))
    L_Acc_Test.append(float(Erms_Test.split(',')[0]))
    
    
 


# In[88]:


print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("Training Accuracy = " + str(np.around(max(L_Acc_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("Validation Accuracy = " + str(np.around(max(L_Acc_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
print ("Testing Accuracy = " + str(np.around(max(L_Acc_Test),5)))

