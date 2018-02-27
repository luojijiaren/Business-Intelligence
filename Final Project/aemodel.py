# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:44:05 2017

@author: fzhan
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

users = pd.read_csv('orders_cutted.csv', engine = 'python', encoding = 'latin-1')
users=users[['order_id','user_id','eval_set']]
prior=pd.read_csv('prior_cutted.csv', engine = 'python', encoding = 'latin-1')
prior=prior[['order_id','product_id']]
train=pd.read_csv('train_cutted.csv', engine = 'python', encoding = 'latin-1')
train=train[['order_id','product_id']] 

'''users = pd.read_csv('data/orders.csv', engine = 'python', encoding = 'latin-1')
users=users[['order_id','user_id','eval_set']]
prior=pd.read_csv('data/order_products__prior.csv', engine = 'python', encoding = 'latin-1')
prior=prior[['order_id','product_id']]
train=pd.read_csv('data/order_products__train.csv', engine = 'python', encoding = 'latin-1')
train=train[['order_id','product_id']]'''

# Getting the number of users and movies
#nb_users = int(max(users.iloc[:,1]))
#nb_products = int(max(prior.iloc[:,1]))
# Getting the number of users and movies
#nb_users = int(max(users.iloc[:,1]))
#nb_products = int(max(prior.iloc[:,1]))

# Converting the data into an array with users in lines and movies in columns
u=sorted(list(set(users.iloc[:,1])))
p=sorted(list(set(prior.iloc[:,1]))+list(set(train.iloc[:,1])))
def convert(data):
    new_data=[]
    n=users.shape[0]
    user_id=0
    i=0
    rec=[]
    pr=[]
    while i<n:
        if users.iloc[i,1]==user_id:
            pr.extend(list(data[data['order_id']==users.iloc[i,0]].iloc[:,1]))
            '''print(users.iloc[i,0])
            #print(pr)
            if len(pr)!=0:
                for j in range(len(pr)):
                    
                    rec[p.index(pr[j])]=1'''
        else:
            #print(user_id)
            #print(pr)
            if len(pr)!=0:
                for j in range(len(pr)):
                    rec[p.index(pr[j])]+=1
            new_data.append(list(rec))
                
            #if sum(rec)!=0:
                #new_data.append(list(rec))
            rec = np.zeros(len(p))
            user_id=users.iloc[i,1]
            pr=list(data[data['order_id']==users.iloc[i,0]].iloc[:,1])
            #print(pr)

        i+=1
    new_data.append(rec)
    new_data=new_data[1:len(u)+2]
    return new_data
'''    for id_users in range(1, users.shape + 1):
        id_orders = users[users.iloc[:,1] == id_users].iloc[:,0]
        rec = np.zeros(1000)
        for oid in range(len(id_orders)):
            id_products=data[data.iloc[:,0]==id_orders.iloc[oid]].iloc[:,1]
            rec[id_products-1]=1

        new_data.append(list(rec))
    return new_data '''
prior_set = convert(prior)
prior_set.to_csv('prior_set.csv',index=False)
train_set = convert(train)
train_set.to_csv('train_set.csv',index=False)
train2=train_set.copy()
prior2=prior_set.copy()

# Converting the data into Torch tensors
prior_set = torch.FloatTensor(prior_set)
train_set = torch.FloatTensor(train_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__() #inheritage
        self.fc1 = nn.Linear(len(p), 220)
        self.fc2 = nn.Linear(220, 10)
        self.fc3 = nn.Linear(10, 220)
        self.fc4 = nn.Linear(220, len(p))
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the SAE
nb_epoch = 80
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.   #float
    for i in range(len(u)):
        #id_user=u[i]
        input = Variable(prior_set[i]).unsqueeze(0)  #add an additional batch dimension at o column
        target = input.clone()   #similar to copy
        if torch.sum(target.data > 0) > 0: #to save space, just include users have rating.
            output = sae(input)
            
            target.require_grad = False #target will not change, save calculation
            output[target == 0] = 0
            
            loss = criterion(output, target)
            mean_corrector = len(p)/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    


# Testing the SAE
test_loss = 0
s = 0.
out=[]
for id_user in range(len(u)):
    input = Variable(prior_set[id_user]).unsqueeze(0)  #use the movies a user has watched(training set)
    target = Variable(train_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        print(output)
        target.require_grad = False
        output[output> 0.5] = 1
        output[output<=0.5]=0
        #out.append([u[i],output])
        l=output.data.numpy()[0]
        
        loss = criterion(output, target)
        #mean_corrector = len(p)/float(torch.sum(target.data > 0) + 1e-10)
        #test_loss += np.sqrt(loss.data[0]*mean_corrector)
        test_loss += np.sqrt(loss.data[0])
        s += 1.
        
print('test loss: '+str(test_loss/s))
#out.to_csv('out.csv')

for i in range(len(out)):
    user=out[i][0]
    