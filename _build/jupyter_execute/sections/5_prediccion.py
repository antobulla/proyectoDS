#!/usr/bin/env python
# coding: utf-8

# # Predicción del precio futuro de ethereum
# 
# 

# ## Regresión lineal 
# En esta sección se desarrolla el primer modelo de predicción: la regresión lineal múltiple, que tiene como función objetivo el error cuadrático medio y se optimiza mediante mínimos cuadrados ordinarios (OLS). 
# 
# Si bien es un modelo simple dentro del abanico de estimadores de machine learning, el método de regresión lineal resulta competitivo en la práctica y presenta ventajas en la facilidad para interpretar los resultados. De esta forma, el trabajo comienza el ejercicio de predicción con OLS para luego pasar a modelos más sofisticados. 
# 
# La sección se estructura de la siguiente manera. En primer lugar, ajustamos y evaluamos la performance del modelo, utilizando una separación básica de train y test siguiendo el enfoque de validación. Luego, realizamos un método de selección conocido como *forward stepwise selection*, que elige los predictores que mayor influencia tengan en la predicción, partiendo de un modelo sin *features* y avanzando uno a uno en variables hasta llegar al modelo con todas las independientes incluidas. Por último, el trabajo realiza la técnica de *k-folds-cross validation* bajo dos formas: la primera, conocida como *Time Series Split Cross-Validation* parte del segmento de observaciones más antiguo para entrenar y evaluar un segmento de observaciones contiguas en el tiempo. El proceso se repite acumulando observaciones al grupo de entrenamiento que en el proceso anterior eran parte del grupo de validación. La segunda forma se conoce como *Blocked Cross-Validation*, la diferencia con la forma anterior radica en que esta no acumula observaciones en entrenamiento, sino que siempre se renuevan los segmentos tanto de *train* como de validación. 
# 
# Por último, el análisis se enfocara en predecir la misma variable *target* pero en momentos distintos del tiempo. En total, son 7 variables a predecir, que van desde el rezago 1 hasta el rezago 7 de la cotización de ethereum. 

# In[1]:


import pandas as pd 
import numpy as np # numpy y pandas to data wrangling 
from datetime import datetime, timedelta # to work w date
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import math


# In[ ]:


### Importamos bases
# train
aux = 'https://drive.google.com/file/d/1JgC5z_ed0YfAjuiYxl9SqqPvgxX3-cZs/view?usp=sharing'
link ='https://drive.google.com/uc?id=' + aux.split('/')[-2]
train = pd.read_csv(link)
train = train.set_index('Date')

# test
aux = 'https://drive.google.com/file/d/1GqALyVt4jKpkUoR2IbQJBvek4Y5rkEg-/view?usp=sharing'
link ='https://drive.google.com/uc?id=' + aux.split('/')[-2]
test = pd.read_csv(link)
test = test.set_index('Date')
# target en train 
aux = 'https://drive.google.com/file/d/1qWQobNkMKTZah0hKhMWiiI1rDy89NaN6/view?usp=sharing'
link ='https://drive.google.com/uc?id=' + aux.split('/')[-2]
y_train = pd.read_csv(link)
y_train = y_train.set_index('Date')
# target en test
aux = 'https://drive.google.com/file/d/1PhiVLaNQ3k7-XifEepwD4LoymDAmixix/view?usp=sharing'
link ='https://drive.google.com/uc?id=' + aux.split('/')[-2]
y_test = pd.read_csv(link)
y_test = y_test.set_index('Date')


# ### Entrenamiento del modelo 

# In[ ]:


#agregar constante
x_train = sm.add_constant(train)
x_test = sm.add_constant(test)


# In[ ]:


# ajuste lineal
model = sm.OLS(y_train.iloc[:,0], x_train.astype(float)).fit()
 
# view model summary
print(model.summary())


# En primer lugar estimamos una regresión lineal entre las predictoras de interés y el precio de ethereum rezagado 1 día, es decir, la cotización del siguiente día. En la salida de regresión se observan variables significativas esperables, como el precio de ethereum y de bitcoin del día corriente y anterior al corriente. En el caso de ethereum, la correlación parcial es positva, interpretándose que un aumento del 1% del precio actual de ethereum se correlaciona, ceteris paribus, con un aumento del 0.99% del precio de ethereum del día siguiente. Mientras que en caso del bitcoin existen dos efectos significativos: el del día corriente y del día anterior al actual. Un aumento del 1% del precio corriente de bitcoin se asocia con una caída del 0.8% del precio de ethereum del día siguiente. No obstante, un aumento del 1% del precio de bitcoin del día anterior al actual se correlaciona con un aumento del 0.84% del valor de ethereum. Resulta interesante como se contraponen ambos efectos. Si lo analizaramos en sentido estático y dejando todo lo demás constante, un aumento del precio de bitcoin en el día actual, seguido de un aumento de la misma criptomoneda en el día anterior, ambas de la misma variación porcentual, se correlacionaría con un leve aumento del precio de ethereum. Ambos efectos jugarían como un rol de sustituto (bitcoin en el día actual) por un lado y complementario (bitcoin del día anterior al actual), por el otro, del valor del siguiente día de ethereum.   
# 
# Resulta también interesante que el ajuste sea de 1, tanto observando el R2 como el R2 ajustado. En base a esto surge la pregunta de si dicho ajuste se mantendría al estimar valores de ethereum más lejanos en el tiempo.  

# In[ ]:


r_squaredAdj = []
for i in range(1,len(y_train.columns)+1):
  model = sm.OLS(y_train.loc[:,f'log_y_lag{i}'], x_train.astype(float)).fit()
  r_squaredAdj = r_squaredAdj + [round(model.rsquared, 4)]  


# In[ ]:


r_squaredAdj


# El bucle anterior extrae de cada regresión entre el valor de ethreum de difentes momentos en el tiempo el R2 ajustado, partiendo desde el primer rezago hasta el séptimo. El ajuste, si bien disminuye al ajustar un modelo que estima momentos más lejanos en el tiempo, sigue siendo considerablemente alto en todas las instancias. 

# ### Evaluación performance en el grupo de entrenamiento
# Primero evaluamos la performance predictiva en el grupo de entrenamiento, siendo conscientes del probable sobreajuste que esta evaluación genere. Para realizar la evluación de desempeño del modelo utilizamos como métrica la raíz del error cuadrático medio como proporción de la media del target. Esto último para poder tener mayor claridad a la hora de interpretar los resultados.

# In[ ]:


# computamos el predicho para cada uno de los rezagos de la dependiente
predictions_dict = {}
for i in range(1,len(y_train.columns)+1):
  model = sm.OLS(y_train.loc[:,f'log_y_lag{i}'], x_train.astype(float)).fit()
  predictions = model.predict()
  predictions_dict[f'pred_y_lag{i}'] = predictions


# In[ ]:


for i in range(1,len(y_train.columns)+1):
  mse = mean_squared_error(y_train.loc[:,f'log_y_lag{i}'], 
                           predictions_dict[f'pred_y_lag{i}'])
  rmse = math.sqrt(mse)
  mean_target_train = y_train.loc[:,f'log_y_lag{i}'].mean()
  print(f"Raíz del error cuadrático medio para log_y_lag{i}:")
  print(f'El error de predicción en el grupo de entrenamiento es equivalente al {round(rmse/mean_target_train*100)}% del promedio de la variable dependiente\n')


# Como es esperable, el ajuste es considerablemente alto, por lo que la predicción también lo es. Aunque la performance disminuye a medida que predecimos valores más lejanos.

# ### Evaluación performance en el grupo de test
# Pasamos a evaluar la performance en el grupo test. 

# In[ ]:


predictions_test_dict = {}
for i in range(1,len(y_test.columns)+1):
  model = sm.OLS(y_train.loc[:,f'log_y_lag{i}'], x_train.astype(float)).fit()
  predictions = model.predict(x_test)
  predictions_test_dict[f'pred_y_lag{i}'] = predictions


# In[ ]:


for i in range(1,len(y_test.columns)+1):
  mse = mean_squared_error(y_test.loc[:,f'log_y_lag{i}'], 
                           predictions_test_dict[f'pred_y_lag{i}'])
  rmse = math.sqrt(mse)
  mean_target_test = y_test.loc[:,f'log_y_lag{i}'].mean()
  print(f"Raíz del error cuadrático medio para log_y_lag{i} en test:")
  print(f'El error de predicción es equivalente al {round(rmse/mean_target_test*100)}% del promedio de la variable dependiente\n')


# Curiosamente, la performance predictiva tiene mejoras en el grupo test respecto a la evaluación en el grupo de entrenamiento. A continuación, pasamos a graficar el observado versus el predicho para cada target estimado.

# In[ ]:


plt.style.use('default')
plt.style.use('ggplot')

fig=plt.figure(figsize=(25,25))
for i in range(0,7):
    ax=fig.add_subplot(4,3, i+1)
    ax.scatter(y_test.loc[:,f'log_y_lag{i+1}'], 
               predictions_test_dict[f'pred_y_lag{i+1}'], 
               edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
    plt.title(f'log_y_lag{i+1}')
    plt.xlabel("valor predicho")
    plt.ylabel("observado")


# Como se puede ver, la dispersión entre el valor real y el estimado aumenta a medida que tratamos de predecir ventanas de tiempo más lejanas al período actual. No obstante, la performance se mantiene alta inclusive a la hora de predecir el valor de ethereum del mismo día de la semana siguiente al actual.

# ### Forward stepwise selection
# 
# Por simplicidad, para el caso de selección y validación cruzada se utilizará solamente el precio de ethereum rezagado siete días. 

# In[ ]:


# x_test_step = sm.add_constant(test)
candidates = list(train.columns)

seleccionado = []
# output_df = pd.DataFrame(d)
train_step = sm.add_constant(train)
test_step = sm.add_constant(test)

# baseline
model_baseline = sm.OLS(y_train.loc[:,'log_y_lag7'], train_step.loc[:,'const'].astype(float)).fit()
# predicción
predictions_test_baseline = model_baseline.predict(test_step.loc[:,'const'])
# métrica
mse_test_baseline = mean_squared_error(y_test.loc[:,'log_y_lag7'], predictions_test_baseline)
rmse_test_baseline = math.sqrt(mse_test_baseline)/y_test.loc[:,'log_y_lag7'].mean()
# df
d = {'var': ["constant"], 'rmse': [rmse_test_baseline]}
output_df = pd.DataFrame(data=d)

# selección
c = 0
while len(candidates) > 0:
  max_rmse = float('inf') 
  for v in candidates:
    # agregar constante
    l =  seleccionado + [v] + ['const']
    # subset
    x_train_step = train_step.loc[:, l]
    x_test_step = test_step.loc[:, l]
    
    # ajuste lineal
    model_step = sm.OLS(y_train.loc[:,'log_y_lag7'], x_train_step.astype(float)).fit()
    # predicción
    predictions_test_step = model_step.predict(x_test_step)
    # métrica
    mse_test_step = mean_squared_error(y_test.loc[:,'log_y_lag7'], predictions_test_step)
    rmse_test_step = math.sqrt(mse_test_step)/y_test.loc[:,'log_y_lag7'].mean()
    if rmse_test_step < max_rmse:
      max_rmse = rmse_test_step
      to_add = v
  candidates.remove(to_add)
  seleccionado.append(to_add)
  output_df = output_df.append(pd.DataFrame({'var': [to_add], 'rmse': [max_rmse]}), ignore_index = True)
  print(output_df)
  c = c + 1


# In[ ]:


output_df


# In[ ]:


mpl.rcParams.update({'font.size': 10}) # set tamaño de ticks
# graficamos la base ethereum completa
# train.index = pd.to_datetime(train.index)
fig, ax = plt.subplots(figsize=(16,10))
ax.plot(output_df.loc[1:,'var'], output_df.loc[1:,'rmse'], color='#993404') # sacamos la constante para mejorar la visibilidad
plt.ylabel("RMSE/media(y)")
plt.xlabel("Predictoras")
plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
plt.title("Performance del modelo bajo selección");


# 

# ### K-folds-cross-validation
# *cross-validation* representa una técnica para validar el modelo separando el conjunto de datos en K grupos, donde para todos estos se realiza entrenamiento y validación. El entrenamiento con todos los datos excepto los pertenecientes a un grupo en cuestión, y se valida la predicción con ese subconjunto dejado afuera. El proceso se repite la cantidad de veces según cantidad de grupos exista. 
# 
# El ejericio se realiza utilizando el subconjunto entrenamiento para la formación de grupos.  

# #### Time Series Split Cross-Validation

# In[ ]:


def date_range(start, end, intv):
    start = datetime.strptime(start,"%Y-%m-%d")
    end = datetime.strptime(end,"%Y-%m-%d")
    diff = (end  - start ) / intv
    for i in range(intv):
        yield (start + diff * i).strftime("%Y-%m-%d")
    yield end.strftime("%Y-%m-%d")


# In[ ]:


x_train.index


# In[ ]:


lower = x_train.index[0]
upper = x_train.index[-1]
timelist = list(date_range(lower, upper, 10))
timelist


# In[ ]:


rmse_matrix_train = []
rmse_matrix_valid = []
for i in range(1, len(timelist)-1):
  first = timelist[0]
  last = timelist[i]
  aux = datetime.strptime(last,"%Y-%m-%d") + timedelta(days=1)
  first_valid = aux.strftime("%Y-%m-%d")
  last_valid = timelist[i+1]
  print(f'Iteration {i} working on range of date {first} to {last} (training) to predict range {first_valid} to {last_valid}')
  x_train_cv = x_train[first:last]
  x_valid_cv = x_train[first_valid:last_valid]
  y_train_cv = y_train.loc[:,'log_y_lag7'][first:last]
  y_valid_cv = y_train.loc[:,'log_y_lag7'][first_valid:last_valid]
  model_cv = sm.OLS(y_train_cv, x_train_cv.astype(float)).fit()
  # Prueba en observaciones usadas para ajustar
  predictions_train_cv = model.predict(x_train_cv)
  mse_train_cv = mean_squared_error(y_train_cv, predictions_train_cv)
  rmse_train_cv = math.sqrt(mse_train_cv)
  mean_target_cv_train = y_train_cv.mean()
  metric_error_train = rmse_train_cv/mean_target_cv_train*100
  rmse_matrix_train.append(metric_error_train)
  # prueba en observaciones del grupo dejado afuera
  predictions_valid_cv = model.predict(x_valid_cv)
  mse_valid_cv = mean_squared_error(y_valid_cv, predictions_valid_cv)
  rmse_valid_cv = math.sqrt(mse_valid_cv)
  mean_target_cv = y_valid_cv.mean()
  metric_error_valid = rmse_valid_cv/mean_target_cv*100
  print(f'  El error de predicción representa el {round(metric_error_train, 2)}% de la media en entrenamiento y el {round(metric_error_valid, 2)}% en validación\n')
  rmse_matrix_valid.append(metric_error_valid)


# In[ ]:


# vemos los resultados en validación para cada iteración
rmse_matrix_valid


# In[ ]:


# promediamos los errores en validación de cada iteración
def Average(lst):
    return sum(lst) / len(lst)

mean_error_cv_valid = Average(rmse_matrix_valid)
mean_error_cv_train = Average(rmse_matrix_train)

print(f'En promedio, la predicción arroja un error del {round(mean_error_cv_train, 2)}% de la media en entrenamiento y el {round(mean_error_cv_valid, 2)}% en validación')


# #### Blocked Cross-Validation

# In[ ]:


rmse_matrix_train_blocked = []
rmse_matrix_valid_blocked = []
for i in range(0, len(timelist)-2):
  first = timelist[i]
  last = timelist[i+1]
  aux = datetime.strptime(last,"%Y-%m-%d") + timedelta(days=1)
  first_valid = aux.strftime("%Y-%m-%d")
  last_valid = timelist[i+2]
  print(f'Iteration {i} working on range of date {first} to {last} (training) to predict range {first_valid} to {last_valid}')
  x_train_cv = x_train[first:last]
  x_valid_cv = x_train[first_valid:last_valid]
  y_train_cv = y_train.loc[:,'log_y_lag7'][first:last]
  y_valid_cv = y_train.loc[:,'log_y_lag7'][first_valid:last_valid]
  model_cv = sm.OLS(y_train_cv, x_train_cv.astype(float)).fit()
  # Prueba en observaciones usadas para ajustar
  predictions_train_cv = model.predict(x_train_cv)
  mse_train_cv = mean_squared_error(y_train_cv, predictions_train_cv)
  rmse_train_cv = math.sqrt(mse_train_cv)
  mean_target_cv_train = y_train_cv.mean()
  metric_error_train = rmse_train_cv/mean_target_cv_train*100
  rmse_matrix_train_blocked.append(metric_error_train)
  # prueba en observaciones del grupo dejado afuera
  predictions_valid_cv = model.predict(x_valid_cv)
  mse_valid_cv = mean_squared_error(y_valid_cv, predictions_valid_cv)
  rmse_valid_cv = math.sqrt(mse_valid_cv)
  mean_target_cv = y_valid_cv.mean()
  metric_error_valid = rmse_valid_cv/mean_target_cv*100
  print(f'  El error de predicción representa el {round(metric_error_train, 2)}% de la media en entrenamiento y el {round(metric_error_valid, 2)}% en validación\n')
  rmse_matrix_valid_blocked.append(metric_error_valid)


# In[ ]:


# promediamos los errores en validación de cada iteración
def Average(lst):
    return sum(lst) / len(lst)

mean_error_cv_valid_blocked = Average(rmse_matrix_valid_blocked)
mean_error_cv_train_blocked = Average(rmse_matrix_train_blocked)

print(f'En promedio, la predicción arroja un error del {round(mean_error_cv_train_blocked, 2)}% de la media en entrenamiento y el {round(mean_error_cv_valid_blocked, 2)}% en validación')


# ### Random Forest

# In[ ]:


# Library
from sklearn.ensemble import RandomForestRegressor
# modelo con 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000)
# entrenamos el modelo
rf.fit(train, y_train['log_y_lag7'])
# Evaluamos el modelo ajustado en test
predictions = rf.predict(test)
## computamos la raíz del error cuadrático promedio sobre la media del target como métrica
# error cuadrático medio
mse = mean_squared_error(y_test['log_y_lag7'], 
                           predictions)
# aplicamos raíz                            
rmse = math.sqrt(mse)
# promedio de la dependiente
mean_target_test = y_test['log_y_lag7'].mean()
# computamos finalmente la métrica
rmse_mean_y = (rmse/mean_target_test)*100 
print(f'En promedio, la predicción en random forest con mil modelos arroja un error del {round(rmse_mean_y, 2)}% en test')



# In[ ]:


## importancia de atributos
# nombre de las variables predictoras
feature_list = list(train.columns)
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[ ]:


# import de las librerías
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# configuro el estilo
plt.style.use('fivethirtyeight')

# defino el tamaño del gráfico
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(5)

# list of x locations for plotting
x_values = list(range(len(importances)))

# Genero el gráfico de barras
plt.bar(x_values, importances, orientation = 'vertical')

# Propiedades de los gráficos, Títulos y etiquetas
plt.xticks(x_values, feature_list, rotation='vertical')
plt.ylabel('Importancia'); plt.xlabel('Variable'); plt.title('Importancia de las variables');


# Al igual que en la regresión lineal, la variable que parece tener más importancia es 'log_price_eth'.

# In[ ]:


# import de las librerías
import graphviz
from sklearn.tree import export_graphviz
import pydot

# Pull out one tree from the forest
tree = rf.estimators_[5]

# Export the image to a dot file
data = export_graphviz(tree, out_file = None, feature_names = feature_list, rounded = True, 
                       precision = 1, filled = True, special_characters = True)

graph = graphviz.Source(data)
graph

