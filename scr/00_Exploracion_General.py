#!/usr/bin/env python
# coding: utf-8

# # Práctica 1: Análisis Exploratorio de datos (EDA)

# El objetivo de esta práctica es ayudar a un banco a tomar mejores decisiones al aprobar préstamos, analizando datos históricos de solicitudes. Es necesario identificar patrones en los clientes que sí cumplen con sus pagos y en los que no, para que el banco pueda evaluar de forma más precisa las futuras solicitudes. La clave es no rechazar a buenos clientes y, al mismo tiempo, reducir los riesgos de que alguien no pague.

# **Problema a resolver**

# La idea principal de este trabajo es analizar los datos de los clientes para encontrar características o patrones que puedan indicar un mayor riesgo de que no paguen un préstamo. Esto servirá para tomar decisiones más acertadas, reduciendo las pérdidas por impagos y evitando rechazar a personas que sí pueden pagar. En las próximas prácticas, voy a crear un modelo de aprendizaje supervisado que ayudará a predecir el riesgo de incumplimiento de forma más precisa.

# **Variables disponibles en producción**

# En producción, se utilizarán únicamente las variables disponibles en el momento en que se solicita el préstamo. Esto incluye datos proporcionados por el cliente, información financiera, demográfica, laboral y cualquier otro dato recopilado antes de decidir si se aprueba o no. No se tomarán en cuenta variables relacionadas con lo que ocurra después de que el préstamo haya sido otorgado.

# **Evaluación del modelo**

# El modelo se evaluará para medir qué tan bien identifica a los clientes solventes y a los de alto riesgo. Esto se llevará a cabo con métricas como la precisión, el recall y la curva ROC-AUC. También se tendrá en cuenta cómo afecta al negocio, especialmente en la reducción de pérdidas por impagos y en la aprobación de clientes confiables. Esta evaluación se realizará en las siguientes etapas del proyecto.

# **Conclusiones esperadas del EDA**

# Al finalizar el EDA, buscamos responder si existe un tipo específico de cliente que sea más propenso a no pagar el préstamo. Este análisis implicará explorar los datos y ver si hay características que están vinculadas a un mayor riesgo de incumplimiento.

# Es importante mencionar que nuestro análisis se desarrolla mediante un código informático que debe ser óptimo y estar correctamente organizado para asegurar claridad y eficiencia.

# In[1]:


# Comprobamos el environment
import sys

print(sys.version)
print(sys.path)
print("---")
print(sys.executable)


# # Exploración general de la tabla

# En este primer notebook se analizarán las variables y se estudiará la distribución y las características de los datos. Así, se verán los tipos de los datos, los valores nulos y las diferentes columnas.

# ## Importo librerias

# Para comenzar, se importan las librerías necesarias en este notebook.

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
from plotnine import ggplot, aes, geom_bar, labs

# Configuramos la visualización en pandas para mostrar hasta 500 columnas 
# y 5000 filas de un DataFrame
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 5000)


# ## Leo la tabla

# El primer paso del trabajo es la lectura de los datos en una tabla, que me servirá como base para explorar y entender las variables disponibles.

# In[3]:


path_folder = "../data/raw/application_data.csv"
pd_loan = pd.read_csv(path_folder)


# In[4]:


pd_loan.head()


# ## Analisis generales de la tabla

# Para realizar un análisis general de la tabla, es fundamental estudiar sus dimensiones y los tipos de datos que contiene.

# Dimensión:

# In[5]:


print(pd_loan.shape)


# In[6]:


# Comprobemos si hay filas duplicadas
print(pd_loan.drop_duplicates().shape)


# Tipos de datos:

# In[7]:


print(list(pd_loan.columns))


# In[8]:


pd_loan.dtypes.to_dict()


# In[9]:


pd_loan.dtypes.sort_values().to_frame('feature_type').groupby(by = 'feature_type').size().to_frame('count').reset_index()


# A partir de la tabla anterior, puedo observar que los datos se dividen en tres tipos: enteros, numéricos de punto flotante y objetos. Como analizaremos más adelante, considero que los datos de tipo float corresponden a variables continuas, mientras que los de tipo object almacenan texto y variables categóricas. Por otro lado, los datos de tipo entero (int) pueden representar tanto variables categóricas (como valores booleanos, por ejemplo) como continuas, dependiendo del contexto.

# ## Exploración de la variable objetivo y tratamiento

# El siguiente paso en nuestro análisis es explorar la variable objetivo, que en este caso es la columna `TARGET`. Esta variable nos indica si el cliente ha incumplido o no con el pago del préstamo, y es crucial para la futura construcción de nuestro modelo.

# Recordamos que la variable objetivo indica si un cliente tiene dificultades de pago (1: retraso superior a X días en al menos una de las primeras Y cuotas del préstamo, 0: en todos los demás casos).

# In[10]:


# Contamos la frecuencia relativa de los valores de la variable objetivo
pd_plot_target = pd_loan['TARGET']        .value_counts(normalize=True)        .mul(100).rename('percent').reset_index()


# Contamos la frecuencia absoluta de los valores de la variable objetivo
pd_plot_target_conteo = pd_loan['TARGET'].value_counts().reset_index()

# Combinamos las tablas anteriores
pd_plot_target_pc = pd.merge(pd_plot_target, pd_plot_target_conteo, on=['TARGET'], how='inner')
pd_plot_target_pc


# In[11]:


pd_plot_target_pc['TARGET'] = pd_plot_target_pc['TARGET'].astype(str)
gg = (
    ggplot(pd_plot_target_pc, aes(x='TARGET', y='percent', weight='percent'))  
    + geom_bar(stat='identity') 
    + labs(x='Variable objetivo', y='Porcentaje', title='Distribución porcentual de la variable objetivo') 
)


# In[12]:


gg


# A partir de la información anterior, se observa que el muestreo está desbalanceado, ya que la mayoría de los clientes no presentan dificultades para pagar el préstamo.

# ## Selección de threshold por filas y columnas para eliminar valores missing

# Analizamos los valores nulos por fila y columna para decidir qué umbral aplicar y eliminar las filas o columnas con demasiados valores faltantes.

# In[13]:


pd_series_null_columns = pd_loan.isnull().sum().sort_values(ascending=False)
pd_series_null_rows = pd_loan.isnull().sum(axis=1).sort_values(ascending=False)
print(pd_series_null_columns.shape, pd_series_null_rows.shape)

pd_null_columnas = pd.DataFrame(pd_series_null_columns, columns=['nulos_columnas'])     
pd_null_filas = pd.DataFrame(pd_series_null_rows, columns=['nulos_filas'])  

pd_null_columnas['porcentaje_columnas'] = pd_null_columnas['nulos_columnas']/pd_loan.shape[0]
pd_null_filas['porcentaje_filas']= pd_null_filas['nulos_filas']/pd_loan.shape[1]


# In[14]:


pd_loan.shape


# In[15]:


pd_null_columnas


# En este trabajo, he establecido un umbral del 80% para eliminar columnas con un exceso de valores faltantes, permitiendo conservar variables útiles y reducir el riesgo de sesgos derivados de imputaciones incorrectas.
# 
# Es crucial recordar que incluso columnas con una alta proporción de valores nulos (por ejemplo, entre un 60% y 70%) podrían contener información relevante que no deberíamos descartar sin una evaluación adecuada.

# In[16]:


threshold=0.8
list_vars_not_null = list(pd_null_columnas[pd_null_columnas['porcentaje_columnas']<threshold].index)
pd_loan_filter_null = pd_loan.loc[:, list_vars_not_null]
pd_loan_filter_null.shape


# In[17]:


pd_null_filas


# Además, en esta sección también podríamos identificar si existen columnas duplicadas en el DataFrame. Esto permitiría detectar redundancias y optimizar el conjunto de datos.

# In[18]:


def duplicate_columns(frame):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función duplicate_columns:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función recibe un DataFrame y busca columnas duplicadas basándose en su contenido. Si dos 
        columnas tienen exactamente los mismos valores, se considera que una de ellas es duplicada.
        
    - Inputs: 
        - frame (DataFrame): DataFrame que contiene las columnas a evaluar.
        
    - Return:
        - dups (list): Lista con los nombres de las columnas duplicadas.
    '''
    
    # Agrupamos las columnas por su tipo de datos
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []
    
    # Recorremos cada grupo de columnas con el mismo tipo de dato
    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)
        
        # Comparamos cada columna con las demás dentro del grupo
        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break
    return dups


# In[19]:


duplicate_cols = duplicate_columns(pd_loan)


# In[20]:


duplicate_cols


# Esto confirma que no existen columnas duplicadas, por lo que no será necesario eliminar ninguna por esa razón.

# ## Tipos: Variables categoricas y continuas

# Como mencioné anteriormente, los datos disponibles se dividen en tres tipos principales: enteros, numéricos de punto flotante y objetos. A partir de esta clasificación inicial, el objetivo es agrupar las variables en dos categorías principales: categóricas y numéricas.
# 
# Las variables categóricas son aquellas que representan grupos o categorías bien definidas, mientras que las numéricas, también conocidas como continuas, son valores que pueden medirse y abarcar un rango amplio, incluso infinito. Esta diferenciación es clave para orientar los análisis posteriores.
# 
# He decidido considerar los datos de tipo punto flotante como continuos, ya que suelen representar valores medibles. Soy consciente de que podría haber excepciones, pero debido a la imposibilidad de revisar cada caso individualmente, tomaré esta decisión inicial y, en caso de errores, los corregiré más adelante.
# 
# En cuanto a los datos de tipo objeto, los clasificaré como categóricos porque suelen representar texto o etiquetas, lo que encaja con la idea de categorías o grupos. Por último, los datos de tipo entero requieren un análisis más detallado. Algunas variables enteras pueden ser booleanas (es decir, tomar solo los valores 0 y 1), lo que las convierte en categóricas, mientras que otras pueden representar cantidades o mediciones y, por tanto, serán tratadas como continuas.

# In[21]:


def dame_variables_categoricas(dataset=None):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función dame_variables_categoricas:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función recibe un DataFrame y devuelve una lista de las variables 
        categóricas (con pocos valores únicos).
        
    - Inputs: 
        - dataset (DataFrame): DataFrame que contiene los datos de entrada.
        
    - Return:
        - lista_variables_categoricas (list): Lista con los nombres de las variables 
          categóricas en el DataFrame.
        - other (list): Lista con los nombres de las variables que no cumplen los criterios 
          para ser categóricas.
        - 1 (int): Indica que la ejecución es incorrecta debido a la falta del 
          argumento 'dataset'.
    '''
    # Verificar que el DataFrame de entrada no sea nulo
    if dataset is None:
        print(u'\nError: Falta el argumento dataset en la función')
        return 1 
    
    lista_variables_categoricas = []  
    other = []  

    # Recorrer las columnas del DataFrame
    for i in dataset.columns:
        
        # Si la columna es de tipo objeto (posiblemente categórica)
        if dataset[i].dtype == object:
            unicos = int(len(np.unique(dataset[i].dropna(axis=0, how='all'))))
            if unicos < 100:
                lista_variables_categoricas.append(i)  
            else:
                other.append(i)  
                
        # Si la columna es de tipo entero                
        if dataset[i].dtype == int:
            unicos = int(len(np.unique(dataset[i].dropna(axis=0, how='all'))))
            if unicos < 20:
                lista_variables_categoricas.append(i)  
            else:
                other.append(i) 

    return lista_variables_categoricas, other


# In[22]:


list_cat_vars, other = dame_variables_categoricas(dataset=pd_loan_filter_null)
# pd_loan_filter_null[list_cat_vars] = pd_loan_filter_null[list_cat_vars].astype("category")
pd_loan_filter_null[list_cat_vars].head()


# In[23]:


list_cat_vars


# In[24]:


other


# Más adelante profundizaremos en el análisis de la lista de variables `other`, que corresponde a variables enteras y continuas.

# A continuación, analizo la distribución de algunas variables categóricas mediante el cálculo de frecuencias absolutas y relativas, lo que me permite entender mejor los patrones presentes en los datos. En este momento, se presentan solo algunos ejemplos representativos; en futuros notebooks se profundizará en el análisis de estas variables con más detalle.

# In[25]:


pd_loan_filter_null['WALLSMATERIAL_MODE'].value_counts()


# In[26]:


pd_loan_filter_null['FONDKAPREMONT_MODE'].value_counts()


# In[27]:


pd_loan_filter_null['NAME_CONTRACT_TYPE'].value_counts() 


# In[28]:


# Contamos la frecuencia relativa de los valores de la variable objetivo
pd_plot_target = pd_loan_filter_null['WALLSMATERIAL_MODE']        .value_counts(normalize=True)        .mul(100).rename('percent').reset_index()


# Contamos la frecuencia absoluta de los valores de la variable objetivo
pd_plot_target_conteo = pd_loan_filter_null['WALLSMATERIAL_MODE'].value_counts().reset_index()

# Combinamos las tablas anteriores
pd_plot_target_pc = pd.merge(pd_plot_target, pd_plot_target_conteo, on=['WALLSMATERIAL_MODE'], how='inner')
pd_plot_target_pc


# Tras un análisis general inicial, concluyo que no es necesario reprocesar las variables de tipo objeto, ya que no presentan porcentajes problemáticos, no mezclan valores categóricos con numéricos, ni muestran otras características que requieran tratamiento adicional.
# 
# Además, decido no guardar el DataFrame final, ya que no se han realizado modificaciones respecto a los datos originales.
