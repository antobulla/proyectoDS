#!/usr/bin/env python
# coding: utf-8

# # Preparación de base google trends
# 
# Google permite a todos los usuarios acceder y procesar datos anonimizados sobre volúmenes de búsquedas con google trends. Los datos representan tendencias de búsqueda y volumen o interés de búsqueda en el tiempo. Los mismos pueden ser filtrados por varios criterios, tales como ubicación geográfica (países, regiones, continentes, etc.), temáticas o temas (deportes, finanzas, multimedia, salud, etc) y diferentes unidades de tiempo (anuales, mensuales, semanales y diarios).
# 
# Google provee el volumen de búsqueda para una palabra indexado entre 0 y 100, donde cero indica el interés de búsqueda relativo más bajo y 100 el más alto dentro del rango de tiempo seleccionado. Esto último es importante remarcar porque el valor del índice de una palabra puede cambiar dependiendo del rango de tiempo utilizado, es decir, dado un intervalo de tiempo, google fija el máximo valor de la serie en 100, y los demás datos se calculan como una tasa respecto a ese valor máximo. Adicionalmente, Google tiene una restricción de unidad de tiempo para intervalos de tiempo específicos. Por ejemplo, una solicitud de una palabra para un mes específico de un año devolverá datos diarios; todo un año retornará datos semanales; mientras que un pedido de más de cinco años de intervalo arrojará datos mensuales. Esto trae una complicación adicional, debido a que intentamos obtener información de frecuencias de búsqueda diarias para un período mayor a 5 años. 
# 
# Para resolver estos problemas relacionado al intervalo de tiempo, seguimos el método de ajuste de datos históricos explicados en [**medium**](https://medium.com/@bewerunge.franz/google-trends-how-to-acquire-daily-data-for-broad-time-frames-b6c6dfe200e6). La idea detrás de ésta técnica es armar primero una serie diaria de una palabra, realizando solicitudes con el intervalo de tiempo máximo para que google arroje valores diarios (pedidos de un rango de 90 días). En paralelo importar la serie histórica de esa misma palabra, es decir, los valores de todo el período bajo análisis, lo cual hará que google nos brinde frecuencias mensuales. Con ambos inputs, los datos ajustados se calculan de la siguiente manera: 
# 
# $$
# gtd_{d-m}*(gtd_{m}/100)
# $$ 
# 
# Donde $gtd$ representa el valor de google trends de un término de búsqueda o tópico específico, que puede ser diario ($d-m$) o mensual ($m$). De esta manera, el valor diario es multiplicado por el valor del mismo mes en cuestión pasado previamente a tasa (dividiendo por 100). Esto genera que el valor máximo local, es decir, de cada intervalo de tiempo diario importado se ajuste por su valor histórico y así obtendríamos los valores diarios ajustados a su frecuencia histórica.
# 
# Para armar la solicitud de términos de búsqueda se utiliza la librería `pytrends`, cuya forma de utilización se puede encontrar en [***hackernoon***](https://hackernoon.com/how-to-use-google-trends-api-with-python). Con esta función creamos nuestra propia función custom `daily_gt()` que sigue la lógica de normalización explicada antes. 

# In[1]:


# instalamos pytrends e importamos algunas librerías que nos serán de utilidad
get_ipython().system('pip install pytrends')
from functools import partial, reduce # to transform variable dic in a dataframe  
import pandas as pd
import time
import pandas as pd 
import numpy as np # numpy y pandas to data wrangling 
from datetime import datetime, timedelta # to work w date
from pytrends.request import TrendReq


# In[2]:


def daily_gt(keyword, start, end, inputCategories ,inputCategoriesNames , hl='en-US', tz=360):
  # función para dividir rango de fechas en segmentos 
  def date_range(start, end, intv):
      start = datetime.strptime(start,"%Y-%m-%d")
      end = datetime.strptime(end,"%Y-%m-%d")
      diff = (end  - start ) / intv
      for i in range(intv):
          yield (start + diff * i).strftime("%Y-%m-%d")
      yield end.strftime("%Y-%m-%d")

  # set pytrends para region y zona horaria
  pytrends = TrendReq(hl=hl, tz=tz) 

  # generación de lista de fechas a utilizar
  firstDate = datetime.strptime(start,"%Y-%m-%d")
  lastDate = datetime.strptime(end,"%Y-%m-%d")
  diffDays_control = lastDate - firstDate 
  if  diffDays_control.days >= 90:
    aux = (lastDate - firstDate)/90
    intv = aux.days
    timelist = list(date_range(start, end, intv))
  else:
    timelist = list([start, end])

  # armamos lista vacía para guardar los resultados
  var_dict={}

  # loop de palabras o categorías a importar
  for x in range(0, len(keyword)):
    varName = keyword[x]
    print(f'{x}: {varName}')
    dataset = pd.DataFrame(columns = [varName])
    # loop de rango de fechas sobre cada palabra o categoría
    for i in range(0, len(timelist)-1):
      if timelist[i] != start:
        startAux = datetime.strptime(timelist[i], "%Y-%m-%d") + timedelta(days=1)
        startNew = startAux.strftime("%Y-%m-%d")
      else:
        startAux = datetime.strptime(timelist[i], "%Y-%m-%d")
        startNew = startAux.strftime("%Y-%m-%d")
      print(f'Iteration from {startNew} to {timelist[i+1]}\n')

      if type(varName)==int: # para considerar las categorías por separado
        pytrends.build_payload(kw_list=[''], cat=varName, timeframe=f'{startNew} {timelist[i+1]}') 
        data = pytrends.interest_over_time()
        loc = inputCategories.index(varName)
        catName = inputCategoriesNames[loc] # para renombrar columnas sin nombres en categorías
        data.rename(columns = {f'':f'{catName}'}, inplace = True)

      else:
        pytrends.build_payload(kw_list=[varName], cat=0, timeframe=f'{startNew} {timelist[i+1]}') 
        data = pytrends.interest_over_time()

      if not data.empty: # chequear que la base importada no esté vacía antes de trabajarla, sino pasar a la prox palabra
        data = pd.DataFrame(data.drop(labels=['isPartial'],axis='columns'))
        data['year'] = data.index.year
        data['month'] = data.index.month 
        data['day'] = data.index.day 
        dataset = dataset.append(data)
        del data
        time.sleep(2) # para aumentar el tiempo entre iteración para evitar el eror 429
      else: 
        continue      
    if type(varName)==int:
      dataset = dataset.iloc[:,1:] # elinamos la columna vacía que se genera por importar categorías que no coinciden con el nombre de la primer columna del dataset generado al ppio
      pytrends.build_payload(kw_list=[''], cat=varName, timeframe='all') 
      historical_data = pytrends.interest_over_time()
      historical_data.rename(columns = {f'':f'{catName}_historical'}, inplace = True)
      varName = catName # para que encuentre las columnas para hacer la normalización
    else:
      pytrends.build_payload(kw_list=[varName], cat=0, timeframe='all') 
      historical_data = pytrends.interest_over_time()
      historical_data.rename(columns = {f'{varName}':f'{varName}_historical'}, inplace = True)

    historical_data = pd.DataFrame(historical_data.drop(labels=['isPartial'], axis='columns'))
    historical_data['year'] = historical_data.index.year
    historical_data['month'] = historical_data.index.month 
      
    dataset = pd.merge(dataset, historical_data, on=["year", "month"])
    del historical_data
    dataset[f'{varName}_adjusted'] = dataset[f'{varName}']*(dataset[f'{varName}_historical']/100) 
    var_dict[varName] = pd.DataFrame(dataset)
  return var_dict


# Una vez elaborada la función `daily_gt()` pasamos a armar las listas de palabras según categoría de búsqueda. Las mismas son agrupadas en la lista final `kw_list` que servirá como input para importar los términos de búsqueda y tópicos elegidos.

# In[3]:


crypto = ['cryptocurrency', 'crypto', 'bitcoin', 'bitcoin price', 'ethereum', 'ethereum price']
monetary = ['stock market', 'wall street', 'interest rate', 'fed', 'bankruptcy']
real = ['taxes', 'investment']
policy = ['china', 'united states', 'war', 'russia']
influencers = ['elon musk', 'do kwon']
topics = ['/m/0vpj4_b', '/m/05p0rrx', '/m/0g_fl', '/m/0108bn2x','/m/0965sb', 
          '/g/1214g6vy', '/m/0drqp', '/m/09jx2', '/m/07g82', '/m/0f10yl','/g/11j2cc_qll','/m/061s4']
topicsNames = ['cryptocurrency_top', 'bitcoin_top', 'investment_top', 
               'ethereum_top','exchange_top', 'bankrup_top', 
               'stock_market_top', 'inflation', 'taxes', 'digital_wallet_top', 'covid19', 'pandemic']
categories = [904, 37, 814] # tópicos
categoriesNames = ['future_commodities', 'banking', 'foreign_currency'] 
kw_list = crypto + monetary + policy + topics + influencers + categories
kw_list


# In[4]:


# fijamos el intervalo de tiempo e importamos la lista de palabras armada antes 
start = '2015-08-01' # parece ser el límite inferior de precios de bitcoin en yahoo finance
end = '2022-06-15'
var_dict = daily_gt(keyword = kw_list, start = start, end = end, 
                    inputCategories=categories, inputCategoriesNames=categoriesNames)


# In[ ]:


var_dict.values()


# Las series importadas de google trends están en un diccionario, por lo que transformamos cada una de esas frecuencias en un mismo dataframe con la fecha como índice.

# In[ ]:


from functools import partial, reduce # to transform variable dic in a dataframe  
# merge todos los dataframes en el diccionario var_dict
my_reduce = partial(pd.merge, on=['year','month','day'], how='outer') 
gtrends = reduce(my_reduce, var_dict.values())
# modificamos las columnas con los tópicos según sus respectivos nombres 
for i in range(0, len(topics)): 
  dtcol = [col for col in gtrends.columns if topics[i] in col]
  for ex in dtcol:
    num = len(topics[i])+1
    word = ex[num:None]
    gtrends.rename(columns={ex: f'{topicsNames[i]}_' + word}, inplace=True)
# generamos la variable fecha para tener como índice
gtrends['Date'] = pd.to_datetime(gtrends[["year", "month", "day"]])
gtrends.set_index('Date', inplace=True)
del var_dict
gtrends


# In[ ]:


from google.colab import drive
drive.mount('drive')
gtrends.to_excel('gtrends.xlsx')
get_ipython().system('cp gtrends.xlsx "/content/drive/MyDrive/ds/proyecto/dataset/"')

