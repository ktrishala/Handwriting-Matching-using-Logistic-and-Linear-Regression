import numpy as np
import pandas as pd
import random
import glob

def DatasetFormation(file1,file2,file3,rows,features):
    df_same = pd.read_csv(file1,sep = ',')
    n = sum(1 for line in open(file2)) - 1            #number of records in file (excludes header)
    skip = sorted(random.sample(range(1,n+1),n-rows)) #the 0-indexed header will not be included in the skip list
    df_diff = pd.read_csv(file2,sep = ',', skiprows=skip)  #reading the different pairs file

    df_pair=pd.concat([df_same, df_diff], ignore_index=True)   #concatenating the same and different pairs
    if(features==9):
        df_features = pd.read_csv(file3,sep = ',', usecols=range(1,11))  #reading the features of each image
    else:
        df_features = pd.read_csv(file3,sep = ',')
    #Pulling the features of the first column images
    df_dataset=pd.merge(df_pair,df_features,left_on = 'img_id_A', right_on = 'img_id',how='inner')
    #Pulling the features of the second column images and concatenating them
    df_dataset=pd.merge(df_dataset,df_features,left_on = 'img_id_B', right_on = 'img_id',how='inner')
    df_dataset=df_dataset.drop(columns=['img_id_x', 'img_id_y'])

    #subtrating the features of respective columns and storing them in seperate columns
    for column_a, column_b, i in zip(df_dataset.columns[3:3+features],df_dataset.columns[3+features:],range(features)):
        df_dataset[i] = df_dataset[column_a] - df_dataset[column_b]
    for column in (df_dataset.columns[-features:]):
        df_dataset[column] = df_dataset[column].abs()
    df_dataset = df_dataset.sample(frac=1).reset_index(drop=True) #shuffling the entire dataset
    return df_dataset

def CreateConcatDataset(df_combine_dataset,features):  #creating the features concatenation dataset
    if(features==9):
        df_combine_dataset.to_csv('HumanObservedDatasetwithFeatureConcatenation.csv',index=False,columns=df_combine_dataset.columns[2:2+features+features+1])
    else:
        df_combine_dataset.to_csv('GSCDatasetwithFeatureConcatenation.csv',index=False,columns=df_combine_dataset.columns[2:2+features+features+1])
    print('Concatenated file created!')

def CreateSubtractedDataset(df_combine_dataset,features):  #creating the features subtraction dataset
    cols=['target']
    for column in (df_combine_dataset.columns[-features:]):   #the subtracted features are the last rows in dataframe
        cols.append(column)
    if(features==9):
        df_combine_dataset.to_csv('HumanObservedDatasetwithFeatureSubtraction.csv',index=False,columns=cols)
    else:
        df_combine_dataset.to_csv('GSCDatasetwithFeatureSubtraction.csv',index=False,columns=cols)
    print('Subtracted file created!')

####Dataset processing for Human Observed Dataset####
df_combine_dataset=DatasetFormation('HumanObserved-Features-Data/same_pairs.csv','HumanObserved-Features-Data/diffn_pairs.csv','HumanObserved-Features-Data/HumanObserved-Features-Data.csv',791,9)
CreateConcatDataset(df_combine_dataset,9)
CreateSubtractedDataset(df_combine_dataset,9)

####Dataset processing for GSC Dataset Dataset####
df_combine_dataset=DatasetFormation('GSC-Features-Data/same_pairs.csv','GSC-Features-Data/diffn_pairs.csv','GSC-Features-Data/GSC-Features.csv',71532,512)
CreateConcatDataset(df_combine_dataset,512)
CreateSubtractedDataset(df_combine_dataset,512)
