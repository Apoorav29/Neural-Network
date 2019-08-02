#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
from sklearn import linear_model
import math as math
import pandas as pd
import numpy as np
from random import shuffle
import sys
import matplotlib.pyplot as plt
import copy as copy

file = open('data.csv', 'rt')
lines = csv.reader(file)
data = list(lines)
del data[0]
for row in data:
	# del row[0]
	# del row[len(row)-1]
	# row.insert(len(row), row[0])
	# del row[0]
	for vals in range(len(row)):
		row[vals] = int(row[vals].strip())
data = np.array([np.array(row) for row in data])
ndata = data
file1 = open('data2.csv','rt')
lines=csv.reader(file1)
test=list(lines)
for row in test:
    for vals in range(len(row)):
        row[vals]=int(row[vals].strip())
test = np.array([np.array(row) for row in test])
print(ndata[0])


# In[18]:


def sigm_derivative(z):
    return z*(1-z)


# In[19]:


def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss


# In[20]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[21]:


def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))


# In[ ]:



    


# In[22]:


actual=ndata[:,0]
ndata = np.delete(ndata, np.s_[0], axis=1)
print(ndata[0])
actual1=test[:,0]
test = np.delete(test, np.s_[0], axis=1)
test=test/255.0
ndata=ndata/255.0


# In[ ]:


y=[]
for i in range(len(ndata)):
    temp = np.zeros(10)
    temp[actual[i]]=1
    y.append(temp)
y=np.array([np.array(row) for row in y])
n_layers=4
n_nodes=[128,128]
num_iters=3000
inp=ndata.shape[1]
out=10
lr=0.013
w_arr=[[] for i in range(n_layers)]
a_arr=[[] for i in range(n_layers)]
del_arr=[[] for i in range(n_layers)]
w_arr[0]=np.random.randn(inp, n_nodes[0])
for i in range(1,n_layers-1):
    w_arr[i]=np.random.randn(w_arr[i-1].shape[1],n_nodes[i-1])
w_arr[n_layers-1]=np.random.randn(n_nodes[len(n_nodes)-1], out)
iter_arr = [i for i in range(1,num_iters+1)]
error_arr = []
for i in range(num_iters):
    a_arr[0]=sigmoid(np.dot(ndata,w_arr[0]))
    for j in range(1,n_layers):
        a_arr[j]=sigmoid(np.dot(a_arr[j-1],w_arr[j]))
    error_arr.append(error(a_arr[n_layers-1], y))
    del_arr[n_layers-1]=(a_arr[n_layers-1]-y)/y.shape[0]
    rev_lst = list(reversed(range(n_layers-1)))
    rev_lst1= list(reversed(range(1,n_layers)))
    for j in rev_lst:
        del_arr[j]=np.dot(del_arr[j+1],np.transpose(w_arr[j+1]))* sigm_derivative(a_arr[j])
    for j in rev_lst1:
        w_arr[j] -= lr*np.dot(np.transpose(a_arr[j-1]),del_arr[j])
    w_arr[0]-= lr*np.dot(np.transpose(ndata), del_arr[0])

# predicted=[]
# x = test
# a_arr[0]=sigmoid(np.dot(x,w_arr[0]))
# for j in range(1,n_layers):
#     a_arr[j]=sigmoid(np.dot(a_arr[j-1],w_arr[j]))
# predicted=np.argmax(a_arr[n_layers-1], axis=1)
# correct=0.0
# print(w_arr[2])
# for i in range(len(actual)):
#     if actual1[i]==predicted[i]:
#         correct+=1
w_arr=np.array(w_arr)
w_arr.dump('thetafinal')
# print(correct/float(len(predicted))*100.0)
plt.plot(iter_arr, error_arr, 'ro')
plt.ylabel('Error')
plt.xlabel('No. of Iterations')
plt.show()


# In[ ]:





# In[ ]:




