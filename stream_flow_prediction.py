from __future__ import division
__author__ = 'mustafa s. dogan 9/12/2022'
import numpy as np
import pandas as pd
import os, warnings
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
warnings.simplefilter('ignore')

# Single Station Data
data_file='1543'
title='Single Station'

# Multi Station Data
# data_file='multi_AGI'
# title='Multi Station'

print('Model: '+title)

print('reading input observed data')
# Read organized data. Organize data in a way that last column is labels and rest is attributes
dataRaw = pd.read_csv('input_dataset_'+data_file+'.csv', header=0)


# Last column label is data type
data_type = dataRaw.keys()[-1]

# ignore first column 
data = dataRaw.iloc[:,1:len(dataRaw.columns)]

# # data summary (you need to print to see results - below)
summary = data.describe()

# Number of rows and columns in data
nDataRow = len(data.index)
nDataCol = len(data.columns) 

# Separate attributes and labels
xList = []
labels = []
Names = data.keys()

# define a lower and upper bound for data if you want to exclude some values
lb = -100000000
ub = 100000000
# separate attributes (xlist)-predictors and labels (labels)- prediction
for i in range(nDataRow):
    dataRow = list(data.iloc[i,0:nDataCol])
    if lb < dataRow[nDataCol-1] < ub:
        xList.append(dataRow[0:nDataCol-1])
        labels.append(float(dataRow[nDataCol-1]))

# number of rows and columns in attributes
nrows = len(xList)
ncols = len(xList[0])

# store attributes (Xlist) and labels (y) in numpy arrays
X = np.array(xList)
y = np.array(labels)
Names = np.array(Names)

random_state = 10101010 # number to fix random seed

# # ******* Data Properties *******

print('visualizing input data')

# if directory to save model outputs do not exist, create one
save_path = 'results_'+data_file
try:
  os.makedirs(save_path)
except OSError:
  pass

# take fixed holdout set 33% of data rows
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=random_state)

# ******* Random Forest Model *******
print('running *Random Forest* model')
# cross-validated parameters
n_estimators_RF=90
max_depth_RF=None
max_features_RF=None

RFModel = ensemble.RandomForestRegressor(n_estimators=n_estimators_RF,
 	                                    max_depth=max_depth_RF, 
 	                                    max_features=max_features_RF,
 	                                    bootstrap=True,
 	                                    # n_jobs = -1, # if n_jobs=-1, all cores are used
 	                                    random_state=random_state)
RFModel.fit(xTrain,yTrain)
predictionRF = RFModel.predict(xTest)

print('reading input prediction testdata')
# Read organized data. Organize data in a way that last column is labels and rest is attributes
dataRaw_p = pd.read_csv('input_dataset_'+data_file+'_predict.csv', header=0)

# Last column label is data type
data_type_p = dataRaw_p.keys()[-1]

# ignore first column 
data_p = dataRaw_p.iloc[:,1:len(dataRaw_p.columns)]

# Number of rows and columns in data
nDataRow_p = len(data_p.index)
nDataCol_p = len(data_p.columns) 

# Separate attributes and labels
xList_p = []
labels_p = []

# define a lower and upper bound for data if you want to exclude some values
lb = -100000000
ub = 100000000
# separate attributes (xlist)-predictors and labels (labels)- prediction
for i in range(nDataRow_p):
    dataRow_p = list(data_p.iloc[i,0:nDataCol_p])
    if lb < dataRow_p[nDataCol_p-1] < ub:
        xList_p.append(dataRow_p[0:nDataCol_p-1])
        labels_p.append(float(dataRow_p[nDataCol_p-1]))

# number of rows and columns in attributes
nrows_p = len(xList_p)
ncols_p = len(xList_p[0])

# store attributes (Xlist) and labels (y) in numpy arrays
X_p = np.array(xList_p)
y_p = np.array(labels_p)

print('predicting prediction testdata')
predictionRF_p = RFModel.predict(X_p)

prediction_save=pd.DataFrame(index=dataRaw_p.iloc[:,0])
prediction_save.index = pd.to_datetime(prediction_save.index)
prediction_save['Test']=y_p
prediction_save['Prediction']=predictionRF_p
prediction_save.to_csv(save_path+'/prediction.csv', index=True)


print('Completed!')