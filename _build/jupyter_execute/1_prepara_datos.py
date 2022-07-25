#!/usr/bin/env python
# coding: utf-8

# **Pendientes desarrollo**
# - Armar un colab aparte para introducir la problemática y el objetivo del ejercicio de predicción
# - Compilar todo una vez terminado en un *Github bookdown*

# ## Preparación dataset

# In[1]:


### Cargamos librería 
get_ipython().system('pip install --upgrade xlrd # to read files .xlsx')
get_ipython().system('pip install yfinance # install yahoo finance library')
import pandas as pd 
import numpy as np # numpy y pandas to data wrangling 
from datetime import datetime, timedelta # to work w date
import yfinance as yf # to import some others data as cryptocurrencies or stocks
from functools import partial, reduce # to transform variable dic in a dataframe  


# ### Google Trends
# 
# Google trends es una herramienta creada en 2006 por [***Google***](https://support.google.com/trends/answer/6248105?hl=es&ref_topic=6248052) que permite monitorear la evolución del número de búsquedas realizadas para una determinada palabra clave o tema a lo largo del tiempo. Todo ello basado en la propia base de datos de Google que almacena todas las búsquedas que los usuarios realizan diariamente. La herramienta permite consultar resultados de búsquedas por país o región (en tiempo real, frecuencia horaria, diaria, semanal o mensual), categoría (finanzas, tecnología, etc.) y tipo de búsqueda (web, imágenes, noticias, compras o búsqueda en YouTube). 
# 
# Siguiendo a [***Google***](https://support.google.com/trends/answer/4365533?hl=es), las tendencias de búsquedas se presentan a través de un indicador que se construye a partir del siguiente proceso de normalización:
# -  Cada punto de datos se divide por el total de búsquedas de la región geográfica y el intervalo de tiempo de interés para comparar su popularidad relativa. 
# - A continuación, los números resultantes se escalan a un intervalo del 0 al 100 en función de la proporción de un tema con respecto al total de búsquedas sobre todos los temas.
# 
# Por lo tanto, las regiones que registran el mismo interés de búsqueda de un término no siempre tienen los mismos volúmenes de búsquedas totales.
# 
# La forma en que se construyen los indicadores, y algunas cuestiones relacionadas a las diferentes frecuencias de tiempo, lleva a considerar algunas cuestiones particulares en los datos a la hora de importar las series para todo el período de interés. En [***1_1_googleTrends***](https://colab.research.google.com/drive/16EsslAqhCxdLE7faten7jimdueDAYsXB#scrollTo=N5B37d9opaMa) se explican esos detalles para construir el *dataframe* **gtrends**. No obstante, en esta sección importamos directamente la base previamente creada. 

# In[2]:


aux = 'https://docs.google.com/spreadsheets/d/1tHq6j9qVaNOoEEk5FQPpCAMZ6hP2Bn6C/edit?usp=sharing&ouid=105868423796285576163&rtpof=true&sd=true'
link ='https://drive.google.com/uc?id=' + aux.split('/')[-2]
gtrends = pd.read_excel(link)
gtrends = gtrends.set_index("Date")
gtrends = gtrends.loc[:, ['adjusted' in i for i in gtrends.columns]]


# In[5]:


gtrends


# Como se puede ver, el *dataset* *gtrends* presenta las tendencias de búsquedas diarias de los términos y temas que consideramos relevantes para predecir el precio de la criptomoneda ethereum. Cada palabra o tópico presenta tres indicadores: la base diaria, histórica y diaria ajustada. Sin embargo, para el ejercicio de predicción utilizaremos solamente la base diaria ajustada de cada término. Los detalles sobre los tres tipos de indicadores y el motivo de utilización de la serie ajustada también se explican en [***1_1_googleTrends***](https://colab.research.google.com/drive/16EsslAqhCxdLE7faten7jimdueDAYsXB#scrollTo=N5B37d9opaMa).     

# ### Criptomonedas 
# 
# El siguiente paso es construir la base de criptomonedas. Por un lado, importamos información de ethereum, nuestro target. Por otro lado, también obtenemos información de bitcoin, los cambios relacionados a la cotización de esta pueden estar correlacionados con los cambios en ethereum, por lo que también la incluimos a la lista de predictores. 
# 
# [***CryptoDataDownload***](https://www.cryptodatadownload.com/) es una plataforma que brinda, entre otras cosas, información histórica de la cotización de diferentes criptomonedas a partir de la API Poloniex. Entre ellas, se encuentran las cotizaciones de ethereum y bitcoin. Los datos comprenden el precio de apertura y clausura en un momento del tiempo dado (horario, diario, etc.), el precio más alto y bajo, y el volumen de transacciones.

# In[6]:


### Ethereum
# leo y proceso el archivo con las cotizaciones de ETH en time frame 1 hora
df_eth_1h = pd.read_csv('https://www.cryptodatadownload.com/cdd/Poloniex_ETHUSDT_1h.csv', skiprows=1)

# creo la columna "date_day" con la fecha (sin la hora)
df_eth_1h['date_day'] = pd.to_datetime(df_eth_1h['date']).dt.date

df_eth_1h = df_eth_1h.set_index('date')
df_eth_1h.index = pd.to_datetime(df_eth_1h.index)

# calculo la desviación estandard
df_stddev = df_eth.groupby('date_day')[['open', 'close']].std()
df_stddev.rename(columns = {'open': 'open_stddev', 'close': 'close_stddev'}, inplace = True)


# variable importada de Poloniex Data (incluye valores anteriores a 2017)
ethereum1 = pd.read_csv('https://www.cryptodatadownload.com/cdd/Poloniex_ETHUSDT_d.csv', skiprows=1)
# ethereum1.rename(columns = 'Volume')
ethereum1 = ethereum1.set_index('date')
ethereum1 = ethereum1.sort_index()
ethereum1.index
ethereum1.index = pd.to_datetime(ethereum1.index)
ethereum1.index = [d.date() for d in ethereum1.index]
ethereum1.index.names = ['Date']
ethereum1 = ethereum1.rename(columns={col: col+'_eth' for col in ethereum1.columns})
# promedio cierre y apertura precio ethereum
ethereum1['eth_close_open_mean'] = ethereum1.loc[:,['close_eth',	'open_eth']].mean(axis=1)
# agregamos a la variable el rezago de un día de la variable a predecir  
for i in range(1, 8):
  ethereum1[f'y_lag{i}'] = ethereum1['eth_close_open_mean'].shift(-i)

# Agrega al df ETH la info del desvío estandard del df_stddev, pero éste df tiniinfo solo desde 2019-04-12
ethereum1 = pd.merge(ethereum1, df_stddev, on = ['Date'], how = 'outer')

### Bitcoin
bitcoin = pd.read_csv('https://www.cryptodatadownload.com/cdd/Poloniex_BTCUSDT_d.csv', skiprows=1)
bitcoin = bitcoin.set_index('date')
bitcoin = bitcoin.sort_index()
bitcoin.index = pd.to_datetime(bitcoin.index)
bitcoin.index = [d.date() for d in bitcoin.index]
bitcoin.index.names = ['Date']
bitcoin = bitcoin.rename(columns={col: col+'_btc' for col in bitcoin.columns})


# In[7]:


## Unión criptomonedas
crypto_pol = pd.merge(ethereum1, bitcoin, left_index=True, right_index=True)
del [ethereum1, bitcoin]
crypto_pol = crypto_pol.drop(['symbol_eth','symbol_btc'], axis = 1) 
crypto_pol.index = pd.to_datetime(crypto_pol.index)


# In[ ]:


crypto_pol


# ### Unión inputs 

# In[ ]:


# unión con merge
df = pd.merge(gtrends, crypto_pol, left_index=True, right_index=True)
# reordenar columnas, poniendo el target adelante
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
# visualización
df


# In[ ]:


# pasamos todas las variables a numéricas
df = df.apply(pd.to_numeric)
df.info()


# In[ ]:


# guardar el dataset en carpeta data set del proyecto en google drive
from google.colab import drive
drive.mount('drive')
df.to_csv('input.csv')
get_ipython().system('cp input.csv "/content/drive/MyDrive/ds/proyecto/dataset/"')

