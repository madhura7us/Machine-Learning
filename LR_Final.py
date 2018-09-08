
# coding: utf-8

# In[6]:

import pickle
import gzip
import numpy as np
import math
import sys
import h5py


def design_T(train_y,validate_y,test_y):
    T_train=np.zeros((len(train_y),10))
    T_validate=np.zeros((len(validate_y),10))
    T_test=np.zeros((len(test_y),10))

    for i in range(len(train_y)):
        T_train[i][train_y[i]]=1
    for i in range(len(validate_y)):
        T_validate[i][validate_y[i]]=1
    for i in range(len(test_y)):
        T_test[i][test_y[i]]=1

    return T_train,T_validate,T_test


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
u=pickle._Unpickler(f)
u.encoding='latin1'
train_data, validate_data, test_data = u.load()
f.close()

train_x_temp=train_data[0]
train_y=train_data[1]

validate_x_temp=validate_data[0]
validate_y=validate_data[1]

test_x_temp=test_data[0]
test_y=test_data[1]

hf=h5py.File('USPS.h5','r')
usps_x = (255-255*np.asarray(hf.get('features')))/255
usps_x = 1-np.asarray(hf.get('features'))
usps_y=np.asarray(hf.get('target'))

train_x=train_x_temp
validate_x=validate_x_temp
test_x=test_x_temp

W=np.ones((10,len(train_x[0])))

Y=np.zeros(10)

T_train, T_validate, T_test=design_T(train_y,validate_y,test_y)
print ("STARTED")
ETerm=np.zeros(W.shape)
cnt=0
for z in range(3):
    print ("ITERATION "+str(z+1))
    for i in range(len(train_x)):
        ZM=np.dot(W,train_x[i])+1
        denom = np.sum(np.exp(ZM))
        E=0
        for k in range(10):
            Y[k]=np.exp(ZM[k])/denom
            E-=np.dot(T_train[i][k],np.log(Y[k]))
        cnt+=1
        for j in range(10):
            W[j]=W[j]-0.01*(Y[j]-T_train[i][j])*train_x[i]
    print (E)

count=0.0
for i in range(len(train_x)):
    ZM=np.dot(W,train_x[i])
    denom = np.sum(np.exp(ZM))
    for k in range(10):
        Y[k]=np.exp(ZM[k])/denom
    if np.where(Y==max(Y))[0][0]==np.where(T_train[i]==max(T_train[i]))[0][0]:
        count+=1
print ("TRAINING ACCURACY")
print (count/len(train_x))

count=0.0
for i in range(len(validate_x)):
    ZM=np.dot(W,validate_x[i])
    denom = np.sum(np.exp(ZM))
    for k in range(10):
        Y[k]=np.exp(ZM[k])/denom
    if np.where(Y==max(Y))[0][0]==np.where(T_validate[i]==max(T_validate[i]))[0][0]:
        count+=1
print ("VALIDATION ACCURACY")
print (count/len(validate_x))

count=0.0
for i in range(len(test_x)):
    ZM=np.dot(W,test_x[i])
    denom = np.sum(np.exp(ZM))
    for k in range(10):
        Y[k]=np.exp(ZM[k])/denom
    if np.where(Y==max(Y))[0][0]==np.where(T_test[i]==max(T_test[i]))[0][0]:
        count+=1
print ("TESTING ACCURACY")
print (count/len(test_x))


count=0.0
for i in range(len(usps_x)):
    ZM=np.dot(W,usps_x[i])
    denom = np.sum(np.exp(ZM))
    for k in range(10):
        Y[k]=np.exp(ZM[k])/denom
    if np.where(Y==max(Y))[0][0]==usps_y[i]:#np.where(T_test[i]==max(T_test[i]))[0][0]:
        count+=1
print ("USPS")
print (count/len(usps_x))


# In[ ]:



