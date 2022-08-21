#!/usr/bin/env python
# coding: utf-8

# # Ingeniería de atributos
# 
# En este apartado se trabajará de la siguiente forma. En primer lugar, realizamos las transformaciones necesarias sobre el target. Posteriormente, se realizan las modificaciones sobre las variables predictoras. 
# 
# Dicho esto, pasamos a cargar las librerías que vamos a utilizar y la base de datos.

# In[1]:


import pandas as pd 
import numpy as np # numpy y pandas to data wrangling 
from datetime import datetime, timedelta # to work w date
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from google.colab import drive


# In[4]:


aux = 'https://drive.google.com/file/d/1GWaKdMRk4Fx9qGgBMioNeO3E7_I3bA8k/view?usp=sharing'
link ='https://drive.google.com/uc?id=' + aux.split('/')[-2]
df = pd.read_csv(link)


# In[5]:


df.columns


# ## Target 

# In[4]:


# armado de variable target y conjunto de variables predictoras 
for i in range(1,8):
  df[f'log_y_lag{i}'] = np.log10(df[f'y_lag{i}']+1)
  df = df.drop(columns=[f'y_lag{i}'])
df


# ## Predictores

# ### Info de ethereum y bitcoin

# #### Precios presentes

# In[5]:


# ethereum
df['log_price_eth'] = np.log10(df['close_eth']+1)
# bitcoin
df['log_price_btc'] = np.log10(df['close_btc']+1)


# ##### Prueba de causalidad de granger 
# Para testear si el precio del bitcoin sirve para predecir el precio futuro de ethereum.

# In[6]:


# precio del bitcoin como predictor del precio futuro de ethereum 
t = grangercausalitytests(df[['log_y_lag1', 'log_price_btc']], maxlag=7)


# In[7]:


# calculamos los rezagos de los 6 días anteriores al momento presente 
for i in range(1,8):
  df[f'log_price_btc_lag{i}'] = df['log_price_btc'].shift(i)


# In[8]:


df[['Date','log_price_btc','log_price_btc_lag1','log_price_btc_lag2', 'log_price_btc_lag3', 'log_price_btc_lag4','log_price_btc_lag5','log_price_btc_lag6','log_price_btc_lag7']]


# ##### Prueba de Dickey-Fuller aumentada
# Para testear la existencia de raíz unitaria y comprobar si deberíamos agregar el precio rezagado de la variable dependiente como predictor

# In[9]:


# ethereum
adfuller(df['log_y_lag1'])[1] 


# #### Volumen de transacciones

# In[10]:


df


# In[11]:


df['log_volume_btc-usd'] = np.log10(df['Volume USDT_btc']+1)
df['log_volume_eth-usd'] = np.log10(df['Volume USDT_eth']+1)


# #### Variación interdiaria precios 

# In[12]:


# ethereum
df['var_price_eth'] = df['close_eth'].div(df['close_eth'].shift(1))-1
# bitcoin
df['var_price_btc'] = df['close_btc'].div(df['close_btc'].shift(1))-1


# In[13]:


df = df.dropna()
df


# In[14]:


# guardar el dataset en carpeta data set del proyecto en google drive
drive.mount('drive')
df.to_csv('dataset.csv')
get_ipython().system('cp dataset.csv "/content/drive/MyDrive/ds/proyecto/dataset/"')

