#!/usr/bin/env python
# coding: utf-8

#                                              Simple Linear Regression
# 
# we will see how to use scikit-learn to implement simple linear regression. We have downloaded a dataset that is related to fuel consumption and Carbon dioxide emission of cars. Then, we split our data into training and test sets, create a model using training set, evaluate our model using test set, and finally use model to predict unknown value. 

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('FuelConsumptionCo2.csv')


# In[5]:


df.head() # To see first five rows of the data


# In[6]:


df.describe() # to summarize the data


# In[7]:


column_data=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]


# In[8]:


column_data.head(4)


# In[9]:


column_data_show=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
column_data_show.hist()
plt.show()


# We will plot each of these features vs the Emission, to see how linear is their relation

# In[13]:


plt.scatter(df.FUELCONSUMPTION_COMB,df.CO2EMISSIONS,color='green')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('CO2EMISSIONS')
plt.show()


# In[16]:


plt.scatter(df.ENGINESIZE,df.FUELCONSUMPTION_COMB,color='red')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.show()


# In[18]:


plt.scatter(df.CYLINDERS,df.CO2EMISSIONS,color='blue')
plt.xlabel('CYLINDERS')
plt.ylabel('CO2EMISSIONS')
plt.show()


#  Create Train and Test Data

# In[46]:


from sklearn.model_selection import train_test_split
x_independent_variable = column_data.iloc[:,0].values


# In[47]:


x_independent_variable


# In[48]:


y_dependent_variable=column_data.iloc[:,-1].values


# In[49]:


y_dependent_variable


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(x_independent_variable, y_dependent_variable, test_size = 0.2, random_state = 0)


# In[51]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()


# In[68]:



X_train = sc_X.fit_transform(X_train.reshape(-1,1))


# In[69]:


X_train


# In[71]:


sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))


# In[72]:


y_train


# In[55]:


from sklearn.linear_model import LinearRegression


# In[56]:


regressor = LinearRegression()


# In[58]:


regressor.fit(X_train , y_train)


# In[59]:


print ('Coefficients: ', regressor.coef_)
print ('Intercept: ', regressor.intercept_)


# In[66]:


plt.scatter(X_train,y_train, color='blue')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# In[73]:


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# In[76]:


X_test


# In[ ]:




