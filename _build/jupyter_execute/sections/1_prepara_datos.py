#!/usr/bin/env python
# coding: utf-8

# ## Preparación dataset

# In[1]:


### Cargamos librería 
get_ipython().system('pip install --upgrade xlrd')
get_ipython().system('pip install yfinance')
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
# Si bien la plataforma brinda datos de diferente unidad de tiempo, la misma presenta una limitación en la longitud de tiempo a importar. Pedidos para los últimos 7 días importará frecuencias de búsqueda horarias; pedidos con rangos menores a 9 meses importa series diarias; entre 9 meses y 5 años importa series semanales, y posterior a esto construye datos mensuales. Adicionalmente, google trends construye un indicador cuya frecuencia relativa se realiza utilizando el máximo (desconocido para el usuario) del período solicitado como denominador, por lo que la frecuencia se interpreta en relación a un máximo local, que puede cambiar según cambie el rango de tiempo solicitado. Ambas consideraciones generan una complicación al querer importar datos diarios para un período superior a los 5 años. Para solucionar esto, se crea la función `daily_gt()` que importa datos por períodos menores a 9 meses y los normaliza para ajustar las frecuencias a su valor histórico, siguiendo el método en [***medium***](https://medium.com/@bewerunge.franz/google-trends-how-to-acquire-daily-data-for-broad-time-frames-b6c6dfe200e6). En [***1_1_googleTrends***](https://colab.research.google.com/drive/16EsslAqhCxdLE7faten7jimdueDAYsXB#scrollTo=N5B37d9opaMa) se corre la función y se importan los datos diarios para el período *08/2015-06/2022*, estos es, desde el inicio de ethereum hasta la actualidad. En esta sección importamos directamente la base previamente creada.  

# In[2]:


aux = 'https://docs.google.com/spreadsheets/d/1tHq6j9qVaNOoEEk5FQPpCAMZ6hP2Bn6C/edit?usp=sharing&ouid=105868423796285576163&rtpof=true&sd=true'
link ='https://drive.google.com/uc?id=' + aux.split('/')[-2]
gtrends = pd.read_excel(link)
gtrends = gtrends.set_index("Date")
gtrends = gtrends.loc[:, ['adjusted' in i for i in gtrends.columns]]


# In[16]:


gtrends


# Como se puede ver, el *dataset* *gtrends* presenta las tendencias de búsquedas diarias de los términos y temas que consideramos relevantes para predecir el precio de la criptomoneda ethereum. Cada palabra o tópico presenta tres indicadores: la base diaria, histórica y diaria ajustada. Sin embargo, para el ejercicio de predicción utilizaremos solamente la base diaria ajustada de cada término. Los detalles sobre los tres tipos de indicadores y el motivo de utilización de la serie ajustada también se explican en [***1_1_googleTrends***](https://colab.research.google.com/drive/16EsslAqhCxdLE7faten7jimdueDAYsXB#scrollTo=N5B37d9opaMa).     

# ### Criptomonedas 
# 
# El siguiente paso es construir la base de criptomonedas. Por un lado, importamos información de ethereum, nuestro target. Por otro lado, también obtenemos información de bitcoin, debido a que los cambios en la cotización de esta pueden estar correlacionados con los cambios en ethereum, por lo que también la incluimos a la lista de predictores. 
# 
# [***CryptoDataDownload***](https://www.cryptodatadownload.com/) es una plataforma que brinda, entre otras cosas, información histórica de la cotización de diferentes criptomonedas a partir de la **API Poloniex**. Entre ellas, se encuentran las cotizaciones de ethereum y bitcoin. Los datos comprenden el precio de apertura y clausura en un momento del tiempo dado (horario, diario, etc.), el precio más alto y bajo, y el volumen de transacciones. Para ambas criptomonedas, importamos para el período 08/2015-06/2022. En el caso de ethereum, también promediamos el precio de cierre y apertura diario, nuestra variable a predecir y computamos los primeros 7 rezagos del promedio. Con los rezagos estaríamos teniendo columnas de precios diarios futuros, desde el día siguiente al actual hasta el mismo día de la siguiente semana al día corriente. 

# In[17]:


# ### Ethereum
# variable importada de Poloniex Data (incluye valores anteriores a 2017)
ethereum1 = pd.read_csv('https://www.cryptodatadownload.com/cdd/Poloniex_ETHUSDT_d.csv', skiprows=1)
ethereum1 = ethereum1.set_index('date')
ethereum1 = ethereum1.sort_index()
ethereum1.index
ethereum1.index = pd.to_datetime(ethereum1.index)
ethereum1.index = [d.date() for d in ethereum1.index]
ethereum1.index.names = ['Date']
ethereum1 = ethereum1.rename(columns={col: col+'_eth' for col in ethereum1.columns})
# promedio cierre y apertura precio ethereum
ethereum1['eth_close_open_mean'] = ethereum1.loc[:,['close_eth',	'open_eth']].mean(axis=1)
# agregamos a la variable los rezagos de los 7 días de la variable a predecir  
for i in range(1, 8):
  ethereum1[f'y_lag{i}'] = ethereum1['eth_close_open_mean'].shift(-i)

### Bitcoin
bitcoin = pd.read_csv('https://www.cryptodatadownload.com/cdd/Poloniex_BTCUSDT_d.csv', skiprows=1)
bitcoin = bitcoin.set_index('date')
bitcoin = bitcoin.sort_index()
bitcoin.index = pd.to_datetime(bitcoin.index)
bitcoin.index = [d.date() for d in bitcoin.index]
bitcoin.index.names = ['Date']
bitcoin = bitcoin.rename(columns={col: col+'_btc' for col in bitcoin.columns})


# Al tener ambas bases de datos pasamos a unirlas según el índice que sería en este caso la fecha.

# In[18]:


## Unión criptomonedas
crypto_pol = pd.merge(ethereum1, bitcoin, left_index=True, right_index=True)
del [ethereum1, bitcoin]
crypto_pol = crypto_pol.drop(['symbol_eth','symbol_btc'], axis = 1) 
crypto_pol.index = pd.to_datetime(crypto_pol.index)


# In[19]:


## visualizamos
crypto_pol


# ### Unión inputs 
# 
# Ya con ambas bases cargadas pasamos a unirlos. Por lo que nos quedaría un *dataframe* con información de las dos criptomonedas utilizadas, ethereum y bitcoin (siendo la primera la moneda a predecir), y las frecuencias de tópicos y términos de búsqueda históricas ajustadas de frecuencia temporal diaria. Debido a que todas las variables en el *dataset* deben ser numéricas, pasamos también a transformar todas las columnas en series numéricas, para evitar cualquier problema relacionado al *type*. 

# In[20]:


# unión con merge
df = pd.merge(gtrends, crypto_pol, left_index=True, right_index=True)
# reordenar columnas, poniendo el target adelante
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
# pasamos todas las variables a numéricas
df = df.apply(pd.to_numeric)
df.info()


# In[21]:


# visualización
df


# Ya con la base construida se pasa a guardar el dataset en formato *.csv* con el nombre de *input*. 

# In[22]:


# guardar el dataset en carpeta data set del proyecto en google drive
from google.colab import drive
drive.mount('drive')
df.to_csv('input.csv')
get_ipython().system('cp input.csv "/content/drive/MyDrive/ds/proyecto/dataset/"')

