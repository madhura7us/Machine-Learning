
# coding: utf-8

# In[1]:

import pickle
import gzip
import numpy as np
import math
import sys
import h5py
import matplotlib.pyplot as plt

#Splitting the dataset into required dimensions of Train,Validate and Test
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



def activate(Z):
    
    out=np.zeros(len(Z))
      #SIGMOID
##    for i in range(len(Z)):
##        out[i]=1.0/(1.0+math.exp(-Z[i]))
##    #TANH
##    for i in range(len(Z)):
##        out[i]=(np.exp(2*Z[i])-1)/(np.exp(2*Z[i])+1)
##
##    #RECTIFIED LINEAR UNIT
    for i in range(len(Z)):
        out[i]=max(0,Z[i])


    return out


def derivative(Z):
    out=np.zeros(Z.shape)
    #SIGMOID
##    out=Z*(1-Z)
    #TANH
##    out=(1-Z**2)
    #RECTIFIED LINEAR UNIT
    for i in range(len(Z)):
        if(Z[i]>0):
            out[i]=1
    return out
    
def softmax(Z):
    out=np.zeros(len(Z))
    denom=np.sum(np.exp(Z))
    for i in range(len(Z)):
        out[i]=np.exp(Z[i])/denom
    return out

#Importing and processing dataset
filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
u=pickle._Unpickler(f)
u.encoding='latin1'
train, validate, test = u.load()
f.close()
print( train[0].shape, train[1].shape )
#print( validate[0].shape, validate[1].shape )
#print( test[0].shape, test[1].shape )

#The index is split into target and labels
train_x_temp=train[0]
train_y=train[1]

validate_x_temp=validate[0]
validate_y=validate[1]

test_x_temp=test[0]
test_y=test[1]

train_x=np.zeros((len(train_x_temp),len(train_x_temp[0])+1))
validate_x=np.zeros((len(validate_x_temp),len(validate_x_temp[0])+1))
test_x=np.zeros((len(test_x_temp),len(test_x_temp[0])+1))


for i in range(len(train_x_temp)):
    train_x[i]=np.append(0,train_x_temp[i])
for i in range(len(validate_x_temp)):
    validate_x[i]=np.append(0,validate_x_temp[i])
for i in range(len(test_x_temp)):
    test_x[i]=np.append(0,test_x_temp[i])


W1=np.random.rand(100,785)
W1=W1*2*0.10
W1=W1-0.10
W2=np.random.rand(10,101)
W2=W2*2*0.10
W2=W2-0.10

Y=np.zeros(10)
#USPS FILE IMPORT 
hf=h5py.File('USPS.h5','r')
usps_x_temp = 1-np.asarray(hf.get('features'))
usps_y=np.asarray(hf.get('target'))
usps_x=np.zeros((len(usps_x_temp),len(usps_x_temp[0])+1))
for i in range(len(usps_x_temp)):
    usps_x[i]=np.append(0,usps_x_temp[i])


T_train, T_validate, T_test=design_T(train_y,validate_y,test_y)
for k in range(3):
    print ("TRAINING ITERATION "+str(k+1))
    for i in range(len(train_x)):
        E=0
        ZM=np.dot(W1,train_x[i])
        A=activate(ZM)
        A_update=np.append(1,A)
        ZF=np.dot(W2,A_update)
        Y=softmax(ZF)
        delk=Y-T_train[i]
        temp=np.dot(W2.T,delk)
        derivative_val=derivative(A_update)
        dell=np.multiply(derivative_val,temp)
        W1=W1-0.01*np.asarray(np.asmatrix(dell[1:]).T*np.asmatrix(train_x[i]))
        W2=W2-0.01*np.asarray(np.asmatrix(delk).T*np.asmatrix(A_update))
        for k in range(10):
            E-=np.dot(T_train[i][k],np.log(Y[k]))
    print (E)
count=0.0
for i in range(len(train_x)):

    ZM=np.dot(W1,train_x[i])
    A=activate(ZM)
    A_update=np.append(1,A)
    ZF=np.dot(W2,A_update)
    Y=softmax(ZF)
    if( np.argmax(Y) == np.argmax(T_train[i])):
        count+=1

print ("TRAINING ACCURACY")
print (count/len(train_x))

count=0.0
for i in range(len(validate_x)):
    ZM=np.dot(W1,validate_x[i])
    A=activate(ZM)
    A_update=np.append(1,A)
    ZF=np.dot(W2,A_update)
    Y=softmax(ZF)
    if( np.argmax(Y) == np.argmax(T_validate[i])):
        count+=1

print ("VALIDATION ACCURACY")
print (count/len(validate_x))

count=0.0
for i in range(len(test_x)):

    ZM=np.dot(W1,test_x[i])
    A=activate(ZM)
    A_update=np.append(1,A)
    ZF=np.dot(W2,A_update)
    Y=softmax(ZF)
    if( np.argmax(Y) == np.argmax(T_test[i])):
        count+=1

print ("TESTING ACCURACY")
print (count/len(test_x))
count=0.0
for i in range(len(usps_x)):

    ZM=np.dot(W1,usps_x[i])
    A=activate(ZM)
    A_update=np.append(1,A)
    ZF=np.dot(W2,A_update)
    Y=softmax(ZF)
    if( np.argmax(Y) == usps_y[i]):
        count+=1

print ("USPS")
print (count/len(usps_x))


# In[3]:

plt.plot(k, E)
plt.show()


# In[ ]:



