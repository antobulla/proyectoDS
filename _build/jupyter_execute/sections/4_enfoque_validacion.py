#!/usr/bin/env python
# coding: utf-8

# ## Enfoque de validación
# 
# 

# In[1]:


import pandas as pd 
import numpy as np # numpy y pandas to data wrangling 
from datetime import datetime, timedelta # to work w date
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive


# In[3]:


aux = 'https://drive.google.com/file/d/1GVQaQzc8orPwL9-XSJ8Y2aV9D0XKjw51/view?usp=sharing'
link ='https://drive.google.com/uc?id=' + aux.split('/')[-2]
df = pd.read_csv(link)


# In[6]:


# set fecha como índice 
df = df.set_index('Date')


# In[7]:


df.info()


# In[8]:


list_gtrends = df.columns[(['adjusted' in i for i in df.columns])]
for y in list_gtrends:
  df[f'{y}_log'] = np.log10(df[f'{y}']+1)
df


# In[10]:


# Creación de dataframe con variables de interés + target
# 'rrt_eth', 'sigma_eth', 'rrt_btc', 'sigma_btc' (VER UN POCO MEJOR ESAS VARIABLES)
xvarName = ['log_volume_eth-usd', 'log_volume_btc-usd',  'var_price_eth', 'var_price_btc',
            'log_price_btc','log_price_btc_lag1','log_price_btc_lag2', 'log_price_btc_lag3', 
            'log_price_btc_lag4','log_price_btc_lag5','log_price_btc_lag6', 'log_price_btc', 
            'log_price_eth'] 
y = df.loc[:, (['log_y_lag' in c for c in df.columns])]
X = df.loc[:, (df.columns.isin(xvarName)) | (['adjusted_log' in i for i in df.columns])]
X = X.dropna() # eliminamos la primera observación, debido a que agregamos tasa de variación interdiaria
X


# In[15]:


y


# In[13]:


start = f'{X.index[0]}' 
end = f'{X.index[0]}'
# Enfoque de validación
train = X[start:'2020-12-31']
test  = X['2021-01-01':]
target_train = y[start:'2020-12-31']
target_test  = y['2021-01-01':]
print('Train Dataset:',train.shape)
print('Test Dataset:',test.shape)


# In[14]:


train


# In[16]:


# Exportamos a google drive
drive.mount('drive')
# train
train.to_csv('train.csv')
get_ipython().system('cp train.csv "/content/drive/MyDrive/ds/proyecto/dataset/"')
# test
test.to_csv('test.csv')
get_ipython().system('cp test.csv "/content/drive/MyDrive/ds/proyecto/dataset/"')
# target en train
target_train.to_csv('target_train.csv')
get_ipython().system('cp target_train.csv "/content/drive/MyDrive/ds/proyecto/dataset/"')
# target en test
target_test.to_csv('target_test.csv')
get_ipython().system('cp target_test.csv "/content/drive/MyDrive/ds/proyecto/dataset/"')

