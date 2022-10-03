#!/usr/bin/env python
# coding: utf-8

# # Ingeniería de atributos
# 
# En esta sección pasaremos a realizar las transformaciones de variables a partir de los patrones encontrados en el análisis exploratorio. Para ello, cargamos las librerías que venimos utilizando y el dataset completo. 

# In[1]:


# cargamos las librerías
import pandas as pd 
import numpy as np  
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
from statsmodels.graphics.tsaplots import plot_pacf
from datetime import datetime, timedelta 
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import seaborn as sns
from seaborn import distplot
import scipy as scp
# muteamos algunos warnings esperables que no afectan los outputs de los códigos
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from google.colab import drive


# In[2]:


aux = 'https://drive.google.com/file/d/1GWaKdMRk4Fx9qGgBMioNeO3E7_I3bA8k/view?usp=sharing'
link ='https://drive.google.com/uc?id=' + aux.split('/')[-2]
df = pd.read_csv(link)
df = df.set_index('Date')
df


# In[ ]:


df.columns


# ## Target: precios futuros de ethereum

# En primer lugar, transformamos los siete precios futuros de ethereum calculando su logaritmo de base 10. No obstante, para evitar valores nulos que generen un valor Infinito como resultado, realizamos previamente le sumamos 1 a todos los valores de las series. Esta transformación es monótona y lineal, por lo que no afectarían a los resultados finales, solamente generará que los valores nulo de las series del target sigan siendo nulos.

# In[3]:


# armado de variable target y conjunto de variables predictoras 
for i in range(1,8):
  df[f'log_y_lag{i}'] = np.log10(df[f'y_lag{i}']+1)
  df = df.drop(columns=[f'y_lag{i}'])
df


# ## Predictores

# ### Rezagos del precio de ethereum y bitcoin

# En el caso de los predictores primero trabajamos sobre los precios presentes de bitcoin y ethereum. Aquí de nuevo realizamos una transformación del mismo tipo que con los targets (logaritmo base 10) con una transformación monónota previa (sumar 1). Luego, computamos los 7 rezagos del precio del bitcoin, que encontramos significativo a partir del test de Granger realizada en el análisis univariado. A su vez, los 2 primeros rezagos del precio de ethereum vistos en el análisis de autocorrelación parcial del mismo análisis. 

# In[4]:


# ethereum
df['log_price_eth'] = np.log10(df['close_eth']+1)
# bitcoin
df['log_price_btc'] = np.log10(df['close_btc']+1)


# In[5]:


# calculamos los rezagos de los 7 días anteriores al momento presente del precio de bitcoin 
for i in range(1,8):
  df[f'log_price_btc_lag{i}'] = df['log_price_btc'].shift(i)
# calculamos los rezagos de los 2 días anteriores al momento presente del precio de bitcoin 
for i in range(1,3):
  df[f'log_price_eth_lag{i}'] = df['log_price_eth'].shift(i)


# In[6]:


# visualizamos las nuevas variables creadas
df[['log_price_btc','log_price_btc_lag1','log_price_btc_lag2', 
    'log_price_btc_lag3', 'log_price_btc_lag4','log_price_btc_lag5',
    'log_price_btc_lag6','log_price_btc_lag7']]


# In[7]:


# visualizamos las nuevas variables creadas
df[['log_price_eth','log_price_eth_lag1','log_price_eth_lag2']]


# ### Volumen de transacciones

# En el caso del volumen de transacciones de ambas criptmonedas utilizadas, también realizamos una transformación logarítmica con el mismo procedimiento realizado antes, de manera de mantener valores más comprimidos y evitar que grandes saltos de magnitud generen problemas a los modelos. 

# In[8]:


df['log_volume_btc-usd'] = np.log10(df['Volume USDT_btc']+1)
df['log_volume_eth-usd'] = np.log10(df['Volume USDT_eth']+1)


# ### Variación interdiaria de precios 

# Si bien hemos visto que no existían efectos indirectos de la variación interdiaria del precio del bitcoin en la relación entre el precio presente de ethereum y su valor futuro, creemos conveniente agregar al análisis para contemplar la posibilidad de que exista un patrón directo que podría ser identificado por parte de los modelos. Como también, creamos la variación interdiaria del precio de ethereum por el mismo motivo.

# In[9]:


# ethereum
df['var_price_eth'] = df['close_eth'].div(df['close_eth'].shift(1))-1
# bitcoin
df['var_price_btc'] = df['close_btc'].div(df['close_btc'].shift(1))-1


# ### Shocks pandemia y listado de ethereum 

# Otra de las variables que debemos agregar en el dataset son los dos shocks relevantes que vimos en el análisis multivariado: la aparición del coronavirus y el listado de la criptomoneda ethereum en la plataforma de binance. Como se analizó antes, cada uno de estos shocks resultó tener un efecto indirecto importante al evaluar la relación entre el precio futuro y presente de ethereum, como también, entre predictores como el precio de bitcoin y nuestro target.

# In[10]:


# shock covid-19
df.loc[df.index < '2020-03-12', 'Shock-COVID'] = 0
df.loc[df.index >= '2020-03-12', 'Shock-COVID'] = 1
# listado en binance
df.loc[df.index < '2017-08-17', 'Binance-Listing'] = 0
df.loc[df.index >= '2017-08-17', 'Binance-Listing'] = 1


# ### Términos de búsqueda
# 
# Por último, realizamos las mismas transformaciones logarítmicas que venimos realizan a los términos de búsqueda de google. La razón es similar al volumen de transacciones: el valor en logaritmo lograría una serie con valores más comprimidos, lo cual ayudaría a la hora de realizar las estimaciones.

# In[11]:


list_gtrends = df.columns[(['adjusted' in i for i in df.columns])]
for y in list_gtrends:
  df[f'{y}_log'] = np.log10(df[f'{y}']+1)
df


# Un tipo de transformación que no realizamos es la separación de tendencia y ciclo de las series aproximadamente estacionarias. Creemos que trabajar con las componentes por separado puede producir un resultado positivo en las predicciones. No obstante, su inclusión será dejada para versiones futuras del trabajo.
# 
# Una vez tranformadas todas las variables reducimos el dataset al subconjunto de columnas relevantes. Debido a que todos los predictores fueron transformados, las mismas se ubican en el último segmento de variables en el orden de las columnas del dataset, por lo que al identificar la primera de ese orden (*log_y_lag1*) filtramos desde esa posición hasta el final.

# In[12]:


# identificamos las columnas de interés y las filtramos
colIndex = df.columns.get_loc("log_y_lag1")
df = df.iloc[:, colIndex:]
# eliminamos todos los NA que aparecen a partir de las nuevas transformaciones
df = df.dropna()
df


# In[13]:


# guardar el dataset en carpeta data set del proyecto en google drive
drive.mount('drive')
df.to_csv('dataset.csv')
get_ipython().system('cp dataset.csv "/content/drive/MyDrive/ds/proyecto/dataset/"')

