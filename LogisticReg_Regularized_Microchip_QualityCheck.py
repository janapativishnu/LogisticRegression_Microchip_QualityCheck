# -*- coding: utf-8 -*-
"""
Problem: Implement a regularized logistic Regression model to classify whether a microchip pass quality assurance test (based on two test scores).
Training data as 3 columns (Score in Exam #1, Score in Exam #2, admitted or not)
This data set is from the machine learning course by Prof Andrew Ng, Coursera

As the training data is very small (100), the Accuracy of prediction is around 87.29% (low).
Higher the data size, higher the prediction accuracy. 

@author: Vishnuvardhan Janapati
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation,linear_model
 
# load data
path='C:/Users/Jeyarani/Desktop/Vishnu/Gitub_Uploads/Logistic Regression/'
df=pd.read_csv(path + 'ex2data2.txt',header=None)
df.columns=['X1','X2','Acceptance'] # rename columns

# Plot data
plt.figure()
idx1=df.index[df['Acceptance']==1]
idx2=df.index[df['Acceptance']==0]
plt.plot(df['X1'].loc[idx1],df['X2'].loc[idx1],'+',color='b',label='pass')
plt.plot(df['X1'].loc[idx2],df['X2'].loc[idx2],'o',color='r',label='fail')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend()
plt.show()

# feature mapping
# as the classifier is nonlinear, it is required to add more features to generate a better decision boundary
# features are added upto the power 6

count=len(df.columns) 
for k in range(2,7):
    for i in range(k+1):
        for j in range(k+1):
            if (i+j==k):
                df['X'+str(count)]=df['X1'].pow(i)*df['X2'].pow(j)
                count+=1
    
## Inputs (X) and labels (y) (Score #1, Score #2, and admission status)
y=np.array(df['Acceptance'])

X=np.array(df.drop(['Acceptance'],1))
#
Sscaler=preprocessing.StandardScaler()
Xs=Sscaler.fit_transform(X)

#
# logistic regression model
LogisticR=linear_model.LogisticRegression(C=1e5)
#
LogisticR.fit(Xs,y)

#
#print('------ Logistic Regression------------')
print('Accuracy of Linear Regression Model is ',round(LogisticR.score(Xs,y)*100,2))