#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('FuelConsumptionCo2.csv')


# In[3]:


df.head(3)


# In[31]:


X_IDP = df.loc[:,['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]


# In[32]:


X_IDP.head(2)


# In[33]:


Y_DP = df.iloc[:,-1].values


# In[34]:


Y_DP


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


X_IDP_train, X_IDP_test, Y_DP_train, Y_DP_test = train_test_split(X_IDP, Y_DP, test_size = 0.3, random_state = 0)


# In[37]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
X_IDP = np.asanyarray(X_IDP_train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
Y_DP = np.asanyarray(Y_DP_train[['CO2EMISSIONS']])
regr.fit (X_IDP, Y_DP)


# In[ ]:




