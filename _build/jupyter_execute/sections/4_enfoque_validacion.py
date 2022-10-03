#!/usr/bin/env python
# coding: utf-8

# # Enfoque de validación

# Antes de pasar al ejercicio de predicción creamos los subconjuntos de datos nos van a servir como inputs para realizar la validación de cada modelo, y de esa forma poder evaluar el desempeño de cada estimador ante la predicción de nuevos datos, con el objetivo de minimizar el problema del sobreajuste.
# 
# Dicho esto, primero cargaremos las librerías necesarias y la base modificada en la sección de ingeniería de atributos. 

# In[1]:


import pandas as pd 
import numpy as np # numpy y pandas to data wrangling 
from datetime import datetime, timedelta # to work w date
from google.colab import drive


# In[2]:


aux = 'https://drive.google.com/file/d/1GVQaQzc8orPwL9-XSJ8Y2aV9D0XKjw51/view?usp=sharing'
link ='https://drive.google.com/uc?id=' + aux.split('/')[-2]
df = pd.read_csv(link)
# set fecha como índice 
df = df.set_index('Date')
df


# Con la base cargada, pasamos a identificar las 7 variables target, que corresponden al precio futuro de ethereum de los 7 días futuros inmediatos al corriente. Con estos nombres identificados creamos el dataset *y* que contendrá solamente los targets y *X* las variables predictoras.

# In[3]:


# Creación de dataframe con variables de interés + target
target_list = list(df.loc[:,('log_y_lag' in c for c in df.columns)].columns)
y = df.loc[:, target_list]
X = df.drop(target_list, axis=1)
# visualizamos el dataframe de predictoras
X


# In[4]:


# visualizamos nuestro dataset de targets
y


# Luego, pasamos a dividir ambos dataset entre entrenamiento y test. Para ello determinamos como separador el 31 de diciembre de 2020. De esta forma, las observaciones previas a esta fecha serán los datos de entrenamiento, y las correspondientes a días posteriores serán parte del subconjunto de testeo.

# In[5]:


start = f'{X.index[0]}' 
end = f'{X.index[0]}'
# Enfoque de validación
train = X[start:'2020-12-31']
test  = X['2021-01-01':]
y_train = y[start:'2020-12-31']
y_test  = y['2021-01-01':]
# visualizamos los datasets de predictores
print('Train Dataset:',train.shape)
print('Test Dataset:',test.shape)


# De esta forma, 1956 observaciones se utilizarán para entrenar el modelo, y las restantes 536 serán para la evaluación final de los estimadores.

# In[6]:


# Exportamos a google drive
drive.mount('drive')
# train
train.to_csv('train.csv')
get_ipython().system('cp train.csv "/content/drive/MyDrive/ds/proyecto/dataset/"')
# test
test.to_csv('test.csv')
get_ipython().system('cp test.csv "/content/drive/MyDrive/ds/proyecto/dataset/"')
# target en train
y_train.to_csv('y_train.csv')
get_ipython().system('cp y_train.csv "/content/drive/MyDrive/ds/proyecto/dataset/"')
# target en test
y_test.to_csv('y_test.csv')
get_ipython().system('cp y_test.csv "/content/drive/MyDrive/ds/proyecto/dataset/"')

