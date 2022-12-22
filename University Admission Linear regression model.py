#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model 
from sklearn.metrics import mean_absolute_error
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("C:/Users/mscta/Downloads/adm_data.csv")


# In[3]:


df.head()


# In[4]:


sns.pairplot(df)


# In[5]:


print('The dimensions of the dataset are ',df.shape, '\n', '-'*100)
print('Total number of duplicate values per row ',df.duplicated().sum(), '\n', '-'*100)
print('NUll values in each column','\n', df.isnull().sum())


# In[6]:


print('Descriptive analysis of data','\n')
df.describe()


# In[7]:


#finding correlation between the data
df.corr()


# In[8]:


plt.figure(figsize=(10,10))
correlation = df.corr()
sns.heatmap(correlation, annot = True, cmap = 'Blues')


# In[9]:


admit_chance = df.corr()['Chance of Admit ']
admit_chance


# In[10]:


admit_chance = pd.DataFrame(admit_chance)
admit_chance


# In[11]:


sns.heatmap(admit_chance, annot = True, cmap = 'Blues')


# In[12]:


x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values
y=y.reshape(-1,1)


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# In[14]:


# Feature scaling (Bringing whole data into a single form)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[15]:


# training the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)


# In[16]:


y_pred = np.round(model.predict(x_test),2)
y_pred


# In[17]:


print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# In[22]:


mae = mean_absolute_error(y_pred, y_test)
mae


# In[ ]:




