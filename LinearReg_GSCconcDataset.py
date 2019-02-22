
# coding: utf-8

# In[ ]:


from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv
import math


# In[ ]:


TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 50
PHI = []


# In[ ]:


def GetTargetVector(filePath):  #creating the numpy array of target labels
    data = pd.read_csv(filePath)
    t=[]
    t=data['target'].values  
    print(np.shape(t))
    return t

def GetRawData(filePath):  #creating the numpy array of input features
    data = pd.read_csv(filePath)
    data = data.loc[:, (data != 0).any(axis=0)]   #deleting columns which are having all zero values
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

def GenerateBigSigma(Data, MuMatrix,TrainingPercent):  #generating the covariance matrix
    BigSigma    = np.zeros((len(Data),len(Data)))    
    DataT       = np.transpose(Data)                 
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    
    for i in range(0,len(DataT[0])):  #looping over the entire column length
        vct = []
        for j in range(0,int(TrainingLen)):   #loopng over the entire row length for 80% data
            vct.append(Data[i][j])    #storing the column values in the list
        varVect.append(np.var(vct))   #calculating the variance of the feature for all datapoints
    print(len(Data))
    for j in range(len(Data)):      
        BigSigma[j][j] = varVect[j]   #storing all the  variances of each feature in the diagonal column of BigSigma                                     #sigma value increases it will only increase the spread of the gaussian curve
    BigSigma = np.dot(200,BigSigma)
    
    return BigSigma

def GetScalar(DataRow,MuRow, BigSigInv):    #calculating the exponential part using the Gaussian radial basis function
    R = np.subtract(DataRow,MuRow)          #formula
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80): #forming the design matrix
    DataT = np.transpose(Data)                         
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix)))               
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):   #so for each feature, M Mu s are generated which will be the center of basis functions
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)     
    return PHI

def GetValTest(TEST_PHI,W):     #generating the output labels for test data using the weights computed
    Y = np.dot(TEST_PHI,W)
    return Y

def GetErms(VAL_TEST_OUT,ValDataAct):  #calculating Erms for error and accuracy
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


# In[ ]:


RawTarget = GetTargetVector('GSCDatasetwithFeatureConcatentation.csv')
RawData = GetRawData('GSCDatasetwithFeatureConcatentation.csv')


# In[ ]:


TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))


# In[ ]:



kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData)) #the clusters and the centroids formed are fitted onto the data points
Mu = kmeans.cluster_centers_  #getting the centroids of the 10 clusters formed to be used for generating design matrix

BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent)
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 10) 
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 10)


# In[ ]:


print(Mu.shape)
print(BigSigma.shape)
print(TRAINING_PHI.shape)
print(TrainingTarget.shape)
print(VAL_PHI.shape)
print(TEST_PHI.shape)


# In[ ]:


W = np.random.normal(0, 1, M).reshape(M)
W_Now=np.dot(abs(W),np.sqrt(1/(M+1))) #Initializing the weight matrix using Xavier’s random weight initialization
La           = 2
learningRate = 0.000005
L_Erms_Val   = []
L_Acc_Val    = []
L_Erms_TR    = []
L_Acc_TR     = []
L_Erms_Test  = []
W_Mat        = []
L_Acc_Test   = []

for i in range(0,400):
    
    Delta_E_D     = np.dot(TRAINING_PHI.transpose(),(np.dot(TRAINING_PHI,W_Now) - TrainingTarget))
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)  #Computing derivative of E from the derivative of the error and the regularizer
    Delta_W       = -np.dot(learningRate,Delta_E) 
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next             #Updating the weights for each iteration based on loss function
    
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next)  #Generating the training labels based on updated weights
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget) #Getting RMS error and accuracy
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    L_Acc_TR.append(float(Erms_TR.split(',')[0]))
    

    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next)      #Generating the validation labels based on updated weights
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct) #Getting RMS error and accuracy
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    L_Acc_Val.append(float(Erms_Val.split(',')[0]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next)   #Generating the testing labels based on updated weights
    Erms_Test = GetErms(TEST_OUT,TestDataAct)     #Getting RMS error and accuracy
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))
    L_Acc_Test.append(float(Erms_Test.split(',')[0]))
 


# In[ ]:


print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("Training Accuracy = " + str(np.around(max(L_Acc_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("Validation Accuracy = " + str(np.around(max(L_Acc_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
print ("Testing Accuracy = " + str(np.around(max(L_Acc_Test),5)))

