#!/usr/bin/env python
# coding: utf-8

# # Codificación de las variables categoricas y escalado 

# En este tercer y último notebook, he utilizado técnicas para convertir las variables categóricas en formatos numéricos adecuados para su uso en modelos de machine learning. Para ello, he empleado métodos como Target Encoding y One-Hot Encoding, según la naturaleza de cada variable. Además, realicé un proceso de escalado en las variables numéricas para asegurarme de que todas las características tengan el mismo rango de valores. Este paso es especialmente importante para modelos que son sensibles a las escalas de las variables, como los algoritmos basados en distancias.

# ## Importo librerias

# Para comenzar, se importan las librerías necesarias en este notebook.

# In[1]:


import pandas as pd 
import numpy as np
import sklearn
# conda install category_encoders
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# ## Lectura de datos del procesado inicial

# Es importante señalar que, en este tercer notebook, trabajaremos con el archivo final generado a partir del procesamiento llevado a cabo en el notebook anterior donde ya habiamos realizado una separación en train y test estratificado. 

# In[2]:


pd_loan_train = pd.read_csv("../data/interim/train_pd_data_preprocessing_missing_outlier.csv")                  .set_index("SK_ID_CURR") 
pd_loan_test = pd.read_csv("../data/interim/test_pd_data_preprocessing_missing_outlier.csv")                  .set_index("SK_ID_CURR") 


# In[3]:


pd_loan_train.columns


# In[4]:


pd_loan_train.dtypes


# ## Codificación de la variable objetivo

# Es importante recordar que la variable objetivo `TARGET` toma el valor 1 para clientes con dificultades de pago (retrasos mayores a X días en al menos una de las primeras Y cuotas) y 0 para todos los demás casos.

# In[5]:


pd_loan_train['TARGET'].value_counts()


# In[6]:


X_train = pd_loan_train.drop('TARGET',axis=1)
X_test = pd_loan_test.drop('TARGET',axis=1)
y_train = pd_loan_train['TARGET']
y_test = pd_loan_test['TARGET']


# ## Codificación del resto de variables categoricas

# En este trabajo, elijo codificar las variables categóricas tipo string de dos maneras: las variables con pocas categorías (tres o menos) se codifican utilizando One-Hot Encoding, mientras que aquellas con más de tres categorías se codifican mediante Target Encoding.
#     
# El enfoque de codificación se basa en la cantidad de categorías: One-Hot Encoding es ideal para pocas categorías, ya que evita suposiciones sobre relaciones, mientras que Target Encoding es más eficiente para muchas categorías, al reducir dimensionalidad y capturar patrones relevantes con la variable objetivo.

# In[7]:


list_columns_cat = list(X_train.select_dtypes("object", "category").columns)
list_other = list(set(X_train.columns)-set(list_columns_cat))


# In[8]:


list_columns_cat


# Con este objetivo, decido separar las columnas categóricas que contienen cadenas de texto en función de sus valores únicos, para así poder determinar si utilizaré One-Hot Encoding o Target Encoding en cada caso.

# In[9]:


def separar_por_unicos(df, list_columns_cat):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función separar_por_unicos:
    ----------------------------------------------------------------------------------------------------------
    - Descripción: 
        Función que recibe un DataFrame y una lista de columnas categóricas y separa las 
        columnas en dos listas según el número de valores únicos que tienen. 
        
    - Inputs:
        - df (DataFrame): Pandas DataFrame que contiene los datos.
        - list_columns_cat (list): Lista con los nombres de las columnas categóricas del dataset.
        
    - Return:
        - list_columns_more_three_cat: Lista con los nombres de las columnas 
          categóricas que tienen más de 3 valores únicos.
        - list_columns_less_three_cat: Lista con los nombres de las columnas 
          categóricas que tienen 3 o menos valores únicos.
    ----------------------------------------------------------------------------------------------------------
    '''
    
    list_columns_more_three_cat = []  
    list_columns_less_three_cat = []  
    
    for col in list_columns_cat:
        num_unicos = df[col].nunique()  # Cuenta el número de valores únicos en la columna
        
        if num_unicos > 3:
            list_columns_more_three_cat.append(col)  
        else:
            list_columns_less_three_cat.append(col)  
    
    return list_columns_more_three_cat, list_columns_less_three_cat


# In[10]:


list_columns_more_three_cat, list_columns_less_three_cat = separar_por_unicos(X_train, list_columns_cat)


# In[11]:


list_columns_more_three_cat


# In[12]:


list_columns_less_three_cat


# Comenzamos codificando con One-Hot Encoding, ideal para variables con pocas categorías.

# In[13]:


ohe = ce.OneHotEncoder(cols=list_columns_less_three_cat)
model_ohe = ohe.fit(X_train, y_train)


# In[14]:


model_ohe


# In[15]:


X_train_ohe = model_ohe.transform(X_train, y_train)
X_test_ohe = model_ohe.transform(X_test, y_test)


# In[16]:


X_train_ohe


# 
# Tenemos que tener en cuenta que One-Hot Encoding aumenta la dimensionalidad del conjunto de datos, ya que crea una nueva columna por cada categoría única en la variable.
# 
# Continuamos codificando con Target Encoding, ideal para variables con muchas categorías.

# In[17]:


te = ce.TargetEncoder(cols=list_columns_more_three_cat)  
model_te = te.fit(X_train_ohe, y_train)  


# In[18]:


model_te


# In[19]:


X_train_t = model_te.transform(X_train_ohe) 
X_test_t = model_te.transform(X_test_ohe, y_test)


# In[20]:


X_train_t


# Por otro lado, Target Encoding no aumenta la dimensionalidad, ya que reemplaza cada categoría por la media del objetivo correspondiente, lo que lo hace más eficiente para variables con muchas categorías.

# In[21]:


X_train_t.dtypes.to_dict()


# ## Escalado de las variables

# Una vez que las variables son numéricas, algunos algoritmos requieren que estén escaladas. Esto significa que los valores deben ajustarse para que estén dentro de una misma escala. Para lograrlo, utilicé el StandardScaler de sklearn, que normaliza los datos. Algoritmos como SVM, KNN y Regresión logística son sensibles a la escala de los datos, y si las variables tienen magnitudes muy diferentes, el rendimiento del modelo puede verse afectado.

# In[22]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
model_scaled = scaler.fit(X_train_t)
X_train_scaled = pd.DataFrame(scaler.transform(X_train_t), columns=X_train_t.columns, index=X_train_t.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_t), columns=X_test_t.columns, index=X_test.index)


# In[23]:


X_train_scaled.describe()


# El escalado de variables ha sido exitoso, ya que cada columna presenta una media cercana a 0 y una desviación estándar de 1, lo que facilitará la comparación entre características al situarlas en una misma escala.

# ## Guardado de la tabla

# Ahora, tras un tercer procesamiento y análisis de los datos, se guarda el DataFrame para conservar este nuevo estado proceso y facilitar su uso en futuros modelos. 

# In[24]:


X_train_scaled.to_csv("../data/processed/pd_train_X_data_scaled.csv")
X_test_scaled.to_csv("../data/processed/pd_test_X_data_scaled.csv")


# # Conclusiones

# 
# Antes de comenzar con las conclusiones, quiero aclarar que, aunque es una buena práctica guardar las funciones automatizadas en un archivo externo o notebook auxiliar para importarlas cuando sea necesario, he decidido no hacerlo en esta primera práctica. Considero que esto podría interrumpir el flujo de trabajo y afectar la claridad del análisis, por lo que he optado por definir cada función en el momento en que la necesite. En futuras prácticas, utilizaré un notebook auxiliar para gestionar estas funciones de manera más eficiente.
# 
# En resumen, he realizado un análisis exploratorio del conjunto de datos para entender las variables y su relación con el incumplimiento de pago. He analizado la distribución de las variables numéricas y categóricas utilizando herramientas visuales como histogramas y boxplots. Además, dividí el dataset en subconjuntos de entrenamiento y test, e identifiqué y traté los valores nulos y outliers.
# 
# He examinado las correlaciones entre las variables utilizando métodos estadísticos como Pearson, V de Cramer y WOE, y transformé las variables categóricas mediante One-Hot Encoding y Target Encoding. Finalmente, apliqué técnicas de escalado a las variables numéricas para prepararlas adecuadamente para el desarrollo de modelos predictivos.
# 
# Tras este primer análisis exploratorio, he observado que sí existe un perfil de cliente más propenso a no devolver un préstamo. En general, los clientes con más hijos tienden a enfrentar mayores dificultades para cumplir con los pagos, lo que sugiere que la carga familiar puede afectar la capacidad de pago.
# 
# En cuanto a los tipos de ingresos, los desempleados y aquellos en baja por maternidad presentan un mayor riesgo de impago. Por otro lado, empresarios y estudiantes parecen estar menos expuestos a este riesgo, lo que se refleja en los valores de WOE, que indican una relación inversa con las dificultades de pago.
# 
# La ocupación también es un factor clave: los clientes en trabajos de baja cualificación (como los Low-Skill Laborers) tienen más dificultades para pagar sus préstamos, mientras que aquellos en ocupaciones más cualificadas, como gerentes o contables, presentan un menor riesgo de impago.
# 
# El nivel educativo también influye significativamente, ya que los clientes con secundaria incompleta tienen una mayor probabilidad de impago, en comparación con aquellos con niveles educativos más altos, quienes suelen tener una mayor estabilidad financiera.
# 
# Finalmente, el tipo de préstamo también juega un papel importante. Los préstamos revolventes están asociados con un menor riesgo de impago, mientras que los préstamos en efectivo tienen una relación más débil con las dificultades de pago.
# 
# En resumen, los clientes más propensos a no devolver un préstamo son aquellos con condiciones laborales o económicas desfavorables, como los desempleados, las personas con trabajos de baja cualificación, aquellos con muchos hijos y los que tienen un nivel educativo bajo. Estos factores pueden ser claves para identificar el riesgo de impago en futuros análisis.
# 
# Quiero mencionar que estas hipótesis serán contrastadas en futuras prácticas al desarrollar el modelo de aprendizaje supervisado. No obstante, he podido formularlas gracias al análisis realizado en los notebooks anteriores, mediante gráficas de barras y boxplots, el cálculo de la V de Cramér, y el uso de Weight of Evidence (WOE) e Information Value (IV).
# 
# 
