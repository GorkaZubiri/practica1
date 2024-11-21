#!/usr/bin/env python
# coding: utf-8

# # Valores missing, outlier y correlaciones

# En este segundo notebook se realiza un análisis y preprocesamiento de variables numéricas y categóricas de manera estructurada. 
# 
# Primero, reviso y ajusto los tipos de datos para asegurarme de que cada variable esté en el formato correcto según su naturaleza. Después, divido el conjunto de datos en dos subconjuntos, train y test, para poder evaluar adecuadamente el modelo más adelante.
# 
# Para las variables numéricas, hago un análisis descriptivo utilizando gráficos que ayudan a explorar su distribución y características principales. También examino las correlaciones con los coeficientes de Pearson y Spearman, identifico y gestiono los outliers y, si es necesario, trato los valores faltantes de acuerdo con el contexto de cada caso.
# 
# En cuanto a las variables categóricas, trabajo con la imputación de los valores faltantes y realizo un análisis de las correlaciones utilizando herramientas como el coeficiente V de Cramer, el Weight of Evidence (WOE) y el Information Value (IV). Estos métodos son útiles para entender mejor las relaciones entre las variables categóricas y las demás variables del conjunto de datos.

# ## Importo librerias 

# Para comenzar, se importan las librerías necesarias en este notebook.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
from sklearn.impute import KNNImputer
import scipy.stats as ss
import warnings
from plotnine import ggplot, aes, geom_bar, labs

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 5000)


# ## Lectura de datos del procesado inicial

# Es importante señalar que, en este segundo notebook, trabajaremos con los datos originales tal como aparecen en el notebook anterior, ya que no he realizado ninguna modificación. Además, se establece la columna `SK_ID_CURR` como índice del DataFrame para facilitar su manipulación y análisis.

# In[2]:


path_folder = "../data/raw/application_data.csv"
pd_loan = pd.read_csv(path_folder).set_index("SK_ID_CURR") 


# Recordamos las dimensiones y los tipos de datos con los que estamos trabajando: 

# In[3]:


pd_loan.shape


# In[4]:


pd_loan.columns


# In[5]:


pd_loan.dtypes.sort_values().to_frame('feature_type').groupby(by = 'feature_type').size().to_frame('count').reset_index()


# Se identifican y clasifican las variables del conjunto de datos en categóricas y continuas, asegurándome de que cada una tenga el tipo de dato correcto según lo que representa. Luego, ajusto los tipos de datos para asegurarme de que se puedan interpretar y manipular de manera adecuada en las etapas siguientes del análisis.

# In[6]:


def dame_variables_categoricas(dataset=None):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función dame_variables_categoricas:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función recibe un DataFrame y devuelve una lista de las variables categóricas 
        (con pocos valores únicos).
        
    - Inputs: 
        - dataset (DataFrame): DataFrame que contiene los datos de entrada.
        
    - Return:
        - lista_variables_categoricas (list): Lista con los nombres de las variables 
          categóricas en el DataFrame.
        - other (list): Lista con los nombres de las variables que no cumplen los criterios 
          para ser categóricas.
        - 1: Indica que la ejecución es incorrecta debido a la falta del 
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


# In[7]:


# Identificar las variables categóricas
list_var_cat, other = dame_variables_categoricas(dataset=pd_loan)
pd_loan[list_var_cat] = pd_loan[list_var_cat].astype("category")

# Seleccionar las columnas que contienen datos numéricos continuos
list_var_continuous = list(pd_loan.select_dtypes(['float', 'int']).columns)
pd_loan[list_var_continuous] = pd_loan[list_var_continuous].astype(float)


# Es importante destacar que algunas variables clasificadas inicialmente como enteras son, en realidad, de naturaleza booleana, lo que las convierte en categóricas. Por otro lado, existen variables enteras que corresponden a valores continuos. Por esta razón, lo más seguro es que las variables almacenadas en `other` representan aquellas que, aunque son enteras, se consideran verdaderamente continuas.

# In[8]:


other


# Es interesante destacar que las variables que representan diferencias de tiempo incluyen valores tanto negativos como positivos, ya que están definidas en relación al momento de la solicitud del crédito. Más adelante, se estudiarán con mayor detenimiento.

# In[10]:


pd_loan.dtypes


# ## Separación en train y test estratificado

# Antes de entrenar el modelo, realizamos una separación estratificada de los datos en conjuntos de entrenamiento y prueba, para asegurar que la distribución de las clases en la variable `TARGET` se mantenga proporcional.

# In[11]:


pd_plot_target = pd_loan['TARGET']        .value_counts(normalize=True)        .mul(100).rename('percent').reset_index()

pd_plot_target_conteo = pd_loan['TARGET'].value_counts().reset_index()
pd_plot_target_pc = pd.merge(pd_plot_target, 
                                  pd_plot_target_conteo, on=['TARGET'], how='inner')


# In[12]:


pd_plot_target_pc['TARGET'] = pd_plot_target_pc['TARGET'].astype(str)
gg = (
    ggplot(pd_plot_target_pc, aes(x='TARGET', y='percent', weight='percent'))  
    + geom_bar(stat='identity')  
    + labs(x='Variable objetivo', y='Porcentaje', title='Distribución porcentual de la variable objetivo')  
)


# In[13]:


gg


# Como hemos comentado en notebook anterior, se puede observar que el muestreo está desbalanceado, ya que la mayoría de los clientes no tienen dificultades para pagar el préstamo.

# El 20% de los datos se asignan al conjunto de prueba y el 80% al conjunto de entrenamiento para entrenar el modelo con la mayoría de los datos y evaluar su desempeño con una muestra representativa y no utilizada en el entrenamiento.

# In[14]:


from sklearn.model_selection import train_test_split
X_pd_loan, X_pd_loan_test, y_pd_loan, y_pd_loan_test = train_test_split(pd_loan.drop('TARGET',axis=1), 
                                                                     pd_loan['TARGET'], 
                                                                     stratify=pd_loan['TARGET'], 
                                                                     test_size=0.2)
pd_loan_train = pd.concat([X_pd_loan, y_pd_loan],axis=1)
pd_loan_test = pd.concat([X_pd_loan_test, y_pd_loan_test],axis=1)


# In[15]:


print('== Train\n', pd_loan_train['TARGET'].value_counts(normalize=True))
print('== Test\n', pd_loan_test['TARGET'].value_counts(normalize=True))


# 
# La salida muestra que tanto en el conjunto de entrenamiento como en el de prueba, las proporciones de ambas clases son prácticamente idénticas, lo que confirma que la separación estratificada ha mantenido el balance de las clases en ambos conjuntos.

# ## Visualización descriptiva de los datos

# 
# Este análisis muestra la cantidad de valores nulos por filas y por columnas en el conjunto de entrenamiento, ayudando a identificar qué variables o registros contienen datos faltantes.

# In[16]:


pd_series_null_columns = pd_loan_train.isnull().sum().sort_values(ascending=False)
pd_series_null_rows = pd_loan_train.isnull().sum(axis=1).sort_values(ascending=False)
print(pd_series_null_columns.shape, pd_series_null_rows.shape)

pd_null_columnas = pd.DataFrame(pd_series_null_columns, columns=['nulos_columnas'])     
pd_null_filas = pd.DataFrame(pd_series_null_rows, columns=['nulos_filas'])  
pd_null_filas['target'] = pd_loan['TARGET'].copy()
pd_null_columnas['porcentaje_columnas'] = pd_null_columnas['nulos_columnas']/pd_loan_train.shape[0]
pd_null_filas['porcentaje_filas']= pd_null_filas['nulos_filas']/pd_loan_train.shape[1]


# In[17]:


pd_null_columnas


# In[18]:


pd_null_filas.head()


# No se ha eliminado ninguna columna en este análisis, ya que no se ha identificado un porcentaje de valores nulos lo suficientemente alto como para justificarlo. Aunque en general es importante eliminar columnas con un exceso de nulos para evitar pérdida de información importante, en este caso todas las columnas se han conservado, considerando que incluso aquellas con valores faltantes podrían seguir siendo relevantes.

# A continuación, visualizamos la distribución de las demás variables en el conjunto de datos, tanto de manera general como en función de la variable objetivo `TARGET`. Esto nos permitirá entender mejor cómo se distribuyen las variables y cómo podrían estar relacionadas con la presencia de dificultades de pago en los clientes (1: dificultades de pago, 0: sin dificultades de pago).

# In[19]:


def plot_feature(df, col_name, isContinuous):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función plot_feature:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función visualiza una variable, mostrando su distribución general y su
        relación con el estado la variable objetivo (TARGET). Para variables continuas,
        se usa un histograma y un boxplot; para variables categóricas, se usa un gráfico 
        de barras y uno de barras apiladas.
        
    - Inputs: 
        - df (DataFrame): DataFrame que contiene los datos de entrada.
        - col_name (str): Nombre de la variable a visualizar.
        - isContinuous (bool): Indica si la variable es continua (True) 
          o categórica (False).
        
    - Return:
         - None: Muestra los gráficos sin devolver valores.
    '''
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    count_null = df[col_name].isnull().sum()
    
    # Gráfico sin considerar la variable objetivo
    if isContinuous:
        sns.distplot(df.loc[df[col_name].notnull(), col_name], kde=False, color='#5975A4', ax=ax1)
    else:
        order = df[col_name].dropna().value_counts().index
        sns.countplot(df[col_name].dropna(), order=order, color='#5975A4', saturation=1, ax=ax1)
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    ax1.set_title(col_name)
    ax1.set_title(col_name+ ' Numero de nulos: '+str(count_null))
    plt.xticks(rotation = 90)

    # Gráfico considerando la variable objetivo
    if isContinuous:
        sns.boxplot(x=col_name, y='TARGET', data=df, ax=ax2, palette='Set2') 
        ax2.set_ylabel('')
        ax2.set_title(col_name + ' by Target')
    else:
        data = df.groupby(col_name)["TARGET"].value_counts(normalize=True).to_frame('proportion').reset_index() 
        data.columns = [col_name, "TARGET", 'proportion']
        #sns.barplot(x = col_name, y = 'proportion', hue= target, data = data, saturation=1, ax=ax2)
        sns.barplot(x = col_name, y = 'proportion', hue= "TARGET", order=order, data = data, saturation=1, ax=ax2, palette='Set2')
        ax2.set_ylabel("TARGET"+' fraction')
        ax2.set_title("TARGET")
        plt.xticks(rotation = 90)
    ax2.set_xlabel(col_name)
    
    plt.tight_layout()
    plt.show()


# Para facilitar la interpretación y comprensión de los gráficos, primero se graficarán las variables continuas y, a continuación, las variables categóricas.

# In[20]:


warnings.filterwarnings('ignore')
for i in list(pd_loan_train.columns):
    if (pd_loan_train[i].dtype==float) & (i!='TARGET'):

        plot_feature(pd_loan_train, col_name=i, isContinuous=True)


# Empiezo analizando las variables continuas. Para ello, me fijo en la forma de la distribución en el histograma, ya que esto me puede indicar si los datos siguen una distribución normal, están sesgados o tienen múltiples picos. También observo el boxplot para identificar valores atípicos y entender la dispersión de los datos a través del rango intercuartílico. Además, examino cómo se relaciona la variable continua con la variable objetivo (`TARGET`), buscando diferencias claras entre las clases. Por último, reviso la cantidad de valores nulos, ya que pueden influir en la calidad del análisis.
# 
# La primera variable que analizo es la población relativa por región (`REGION_POPULATION_RELATIVE`). El boxplot muestra una ligera diferencia en función de la variable objetivo, con un rango intercuartílico más amplio en las regiones sin dificultades de pago. Esto podría sugerir que las regiones con mayor población tienen menos dificultades para cumplir con los pagos.
# 
# La siguiente variable es la edad en días de vida (`DAYS_BIRTH`). El boxplot revela que los clientes más jóvenes tienen más dificultades para pagar, ya que la mediana de este grupo es más baja en comparación con aquellos que no presentan dificultades de pago. 
# 
# También es interesante observar la relación con el coche de los clientes (`OWN_CAR_AGE`), los clientes que poseen coches de mayor antigüedad parecen tener más dificultades para pagar
# 
# Es útil analizar las variables `EXT_SOURCE_` (puntuación normalizada de una fuente de datos externa), cuyos histogramas siguen una distribución normal sesgada, con un pico a la derecha. Los boxplots también muestran medianas diferentes según la variable objetivo. Sin embargo, debido a la falta de información sobre la definición exacta de estas variables, no es posible sacar conclusiones definitivas.
# 
# Otra variable interesante a comentar es `FLOORSMAX_` (información normalizada sobre el edificio donde vive el cliente). Aunque presenta una gran cantidad de valores nulos y outliers, lo que podría afectar la validez de las conclusiones, los valores más bajos parecen estar asociados con mayores dificultades de pago.
# 
# También quiero destacar el impacto del tiempo transcurrido desde que el cliente cambió su número de teléfono antes de solicitar la aplicación (`DAYS_LAST_PHONE_CHANGE`). Se observa que aquellos clientes que realizaron este cambio más recientemente tienden a presentar mayores dificultades para cumplir con sus pagos.
# 
# Por último, es importante destacar que, debido a la distribución de los datos y la forma en que se presentan, es complicado sacar conclusiones claras de algunas variables. Entre ellas se encuentran `YEARS_BEGINEXPLUATATION_AVG` y `NONLIVINGAPARMENTS_AVG`, entre otras.

# In[21]:


warnings.filterwarnings('ignore')
for i in list(pd_loan_train.columns):
    if (pd_loan_train[i].dtype!=float) & (i!='TARGET'):
        # print(i)
        plot_feature(pd_loan_train, col_name=i, isContinuous=False)


# Para analizar las variables categóricas, inicio observando gráficos de barras, ya que son útiles para visualizar la distribución de las categorías y detectar posibles desbalances. Además, empleo gráficos de barras apiladas en función de la variable objetivo (`TARGET`) para analizar cómo se distribuyen las clases de `TARGET` entre las diferentes categorías. Si algunas categorías muestran una relación más marcada con la variable objetivo, esto puede indicar un mayor poder predictivo. También reviso la cantidad de valores nulos en estas variables, ya que su presencia puede afectar la calidad de los datos.
# 
# Es importante tener en cuenta que la variable objetivo está desbalanceada: el 91.93% de los casos corresponden a `TARGET` = 0 y solo el 8.07% a `TARGET`= 1. Este desequilibrio influye en la interpretación de los resultados y debe considerarse al realizar el análisis.
# 
# La primera variable analizada es el sexo del cliente. Aunque las diferencias son ligeras, se observa que los hombres, además de ser quienes solicitan menos préstamos, presentan una mayor probabilidad de enfrentar problemas con los pagos.
# 
# Una variable que destaca es el número de hijos de los clientes (`CNT_CHILDREN`). Los datos revelan que los clientes con más hijos tienden a experimentar mayores dificultades para cumplir con los pagos. Sin embargo, dado que la mayoría de los clientes tiene entre 0 y 1 hijo, las conclusiones sobre aquellos con un número elevado de hijos se basan en una muestra limitada, lo que reduce su representatividad.
# 
# Otra variable relevante es el tipo de ingresos de los clientes (`NAME_INCOME_TYPE`). Los resultados muestran que las personas sin empleo y aquellas en baja por maternidad enfrentan mayores dificultades para pagar. Por el contrario, los clientes que son empresarios no presentan problemas significativos con sus pagos.
# 
# En cuanto a la ocupación, los gráficos revelan que los clientes con trabajos de baja cualificación (Low-Skill Laborers) tienen más dificultades para cumplir con sus pagos. Este hallazgo resalta el impacto de la estabilidad laboral y los niveles de ingreso en el comportamiento de pago.
# 
# La región de residencia también influye significativamente en la probabilidad de dificultades de pago (`REGION_RATING_CLIENT`). En particular, los clientes de la región 3 son los que presentan mayores problemas para cumplir con sus pagos, seguidos por los de la región 2. En contraste, los clientes de la región 1 son quienes tienen menos dificultades.
# 
# Otro aspecto interesante es la variable relacionada con el documento 2 (`FLAG_DOCUMENT_2`). Los datos indican que las personas que han entregado este documento tienen una mayor proporción de dificultades de pago. Sin embargo, dado que la mayoría de los clientes no han proporcionado este documento, esta conclusión podría no ser representativa del comportamiento general.
# 
# Por último, en las variables categóricas booleanas no he identificado información particularmente útil. Esto se debe a que, al tener solo dos valores posibles, su variabilidad es limitada, lo que dificulta la detección de patrones claros o relaciones significativas con la variable objetivo. 
# 
# Es muy importante destacar que los datos de tipo entero que he definido como categóricos, incluidas las variables booleanas, no presentan ningún valor nulo.

# # Tratamiento de las variables continuas

# A continuación, se tratarán tres aspectos clave del análisis de los datos: los valores faltantes (missing), las correlaciones entre las variables continuas y los valores atípicos (outliers), con el objetivo de limpiar y entender mejor los datos antes de construir el modelo.
# 
# Primero, recuerdo cuales son las variables que he considerado como continuas: 

# In[22]:


list_var_continuous


# ## Tratamiento de outliers

# Los valores outliers se pueden sustituir por la media, mediana o utilizando valores extremos como la media ± 3 veces la desviación estándar. 
# 
# Es importante destacar que, antes de tratar los outliers, se debe analizar su relación con la variable objetivo y comprender su contexto, ya que podrían representar casos relevantes o errores de medición que influyan en la predicción del modelo.

# In[23]:


def get_deviation_of_mean_perc(pd_loan, list_var_continuous, target, multiplier):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función get_deviation_of_mean_perc:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función calcula el porcentaje de valores que se encuentran fuera de un 
        intervalo de confianza, determinado por la media y una desviación estándar 
        multiplicada por un factor (multiplier), para cada variable continua en el 
        DataFrame. Luego, analiza la relación entre estos valores atípicos y la variable 
        objetivo (TARGET), y devuelve un resumen con los porcentajes de valores atípicos 
        y su distribución en relación con la variable objetivo.
        
    - Inputs: 
        - pd_loan (DataFrame): DataFrame que contiene los datos de entrada.
        - list_var_continuous (list): Lista con los nombres de las variables continuas 
          a analizar.
        - target (str): Nombre de la variable objetivo en el DataFrame.
        - multiplier (float): Factor multiplicador para calcular el intervalo de confianza
          (desviación estándar).
        
    - Return:
        - pd_final (DataFrame): DataFrame que contiene el porcentaje de valores atípicos 
          por cada variable continua, su distribución con respecto a la variable objetivo 
          (TARGET), y otros detalles relevantes.
    '''
    
    pd_final = pd.DataFrame()
    
    for i in list_var_continuous:
        
        series_mean = pd_loan[i].mean()
        series_std = pd_loan[i].std()
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp
        size_s = pd_loan[i].size
        
        perc_goods = pd_loan[i][(pd_loan[i] >= left) & (pd_loan[i] <= right)].size/size_s
        perc_excess = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size/size_s
        
        if perc_excess>0:    
            pd_concat_percent = pd.DataFrame(pd_loan[target][(pd_loan[i] < left) | (pd_loan[i] > right)]                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0,0], 
                                         pd_concat_percent.iloc[0,1]]

            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_outlier_values'] = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size
            pd_concat_percent['porcentaje_sum_null_values'] = perc_excess
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final


# In[24]:


get_deviation_of_mean_perc(pd_loan_train, list_var_continuous, target='TARGET', multiplier=3)


# Tras el análisis exploratorio, he decidido, como primera iteración, no sustituir los valores atípicos, ya que es importante evaluar su impacto en el modelo. Esta decisión también se debe a la gran cantidad de outliers observados, algunos de los cuales superan el 2% de los datos. Se puede observar que el porcentaje de la variable objetivo en cada variable cambiará al no considerar los outliers. Una vez construido el modelo, puedo realizar iteraciones utilizando diferentes métodos de tratamiento para evaluar si estos mejoran el rendimiento

# ## Correlaciones

# En esta sección se analizarán las correlaciones entre las variables continuas utilizando la matriz de correlación de Pearson, que mide la fuerza y la dirección de una relación lineal entre dos variables numéricas. Los valores de Pearson oscilan entre -1 y 1, donde 1 indica una relación positiva perfecta, -1 una relación negativa perfecta, y 0 señala la ausencia de una relación lineal.
# 
# 

# Es importante mencionar que se establece la autocorrelación en 0 para evitar distracción visual y centrar el análisis en relaciones entre distintas variables

# In[25]:


def get_corr_matrix(dataset = None, metodo='pearson', size_figure=[10,8]):
    ''''
    ----------------------------------------------------------------------------------------------------------
    Función get_corr_matrix:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función calcula y visualiza la matriz de correlación entre las variables 
        numéricas de un conjunto de datos. 

    - Inputs: 
        - dataset (DataFrame): Conjunto de datos con las variables numéricas a analizar.
        - metodo (str): Método de correlación a utilizar.
        - size_figure (list): Tamaño de la figura del gráfico.

    - Return:
        - None: Muestra un mapa de calor de la matriz de correlación.
    ----------------------------------------------------------------------------------------------------------
    '''
    
    # Comprobación de que se ha proporcionado el dataset
    if dataset is None:
        print(u'\nHace falta pasar argumentos a la función')
        return 1
    sns.set(style="white")
    
    # Calcular la matriz de correlación
    corr = dataset.corr(method=metodo) 
    
    # Establecer la autocorrelación a cero para evitar distracciones
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
    
    f, ax = plt.subplots(figsize=size_figure)
    
    # Dibujar el mapa de calor con la correlación
    sns.heatmap(corr, center=0,
                square=True, linewidths=.5,  cmap ='viridis' ) #cbar_kws={"shrink": .5}
    plt.show()
    
    return 0


# In[26]:


get_corr_matrix(dataset = pd_loan_train[list_var_continuous], 
                metodo='pearson', size_figure=[10,8])


# En la matriz de correlación se pueden ver varias correlaciones de Pearson cercanas o iguales a 1, lo que indica que algunas variables están perfectamente correlacionadas. Esto podría ser una señal de que hay variables redundantes o derivadas de otras, lo que podría generar problemas de multicolinealidad en los modelos estadísticos.
# 
# Un patrón interesante es que las variables relacionadas con medidas como la media, mediana y moda de una misma característica muestran correlaciones cercanas a 1. Esto sugiere que estas variables podrían estar representando prácticamente la misma información.
# 
# Más adelante, se revisarán estas variables y, si es necesario, se eliminarán aquellas que sean idénticas o estén altamente correlacionadas para evitar que afecten los resultados del modelo.
# 
# En el caso de algoritmos basados en árboles, como XGBoost y Random Forest, la multicolinealidad no representa un problema mayor, ya que estos modelos no necesitan que las variables sean independientes entre sí para hacer predicciones. De hecho, son capaces de manejar variables correlacionadas de forma eficiente sin que esto afecte su rendimiento.
# 
# Sin embargo, en modelos lineales como los GLM (Modelos Lineales Generalizados), la multicolinealidad sí puede afectar la estabilidad e interpretación de los coeficientes. Esto ocurre porque la alta correlación entre variables puede inflar los errores estándar, generando estimaciones inexactas. En estos casos, será importante eliminar o reducir la colinealidad antes de entrenar el modelo.

# In[27]:


corr = pd_loan_train[list_var_continuous].corr('pearson')
new_corr = corr.abs()
new_corr.loc[:,:] = np.tril(new_corr, k=-1)
new_corr = new_corr.stack().to_frame('correlation').reset_index().sort_values(by='correlation', ascending=False)
new_corr[new_corr['correlation']>0.6]


# En esta primera iteración, he optado por no eliminar las variables con altas correlaciones. Estas podrían ser candidatas a eliminación en pasos posteriores. Si en etapas futuras se utiliza un algoritmo que requiera la eliminación de la colinealidad, se procederá a eliminar las variables más correlacionadas para evitar posibles problemas en los resultados del modelo.

# Por otro lado, comprobar la correlación de Spearman puede ser útil, ya que mide relaciones monótonas no lineales entre variables. Esto nos permitirá identificar patrones que podrían ser pasados por alto con Pearson.

# In[28]:


get_corr_matrix(dataset = pd_loan_train[list_var_continuous], 
                metodo='spearman', size_figure=[10,8])


# In[29]:


corr = pd_loan_train[list_var_continuous].corr('spearman')
new_corr = corr.abs()
new_corr.loc[:,:] = np.tril(new_corr, k=-1) # below main lower triangle of an array
new_corr = new_corr.stack().to_frame('correlation').reset_index().sort_values(by='correlation', ascending=False)
new_corr[new_corr['correlation']>0.6]


# La matriz de correlación de Spearman muestra relaciones monótonas, ya sean lineales o no lineales, y tiene la ventaja de ser menos sensible a los outliers. Por otro lado, la matriz de correlación de Pearson está enfocada en relaciones lineales. Ambas matrices brindan información útil sobre las dependencias entre las variables.
# 
# Al comparar ambas matrices, se puede ver que las relaciones entre las variables son bastante similares tanto en Pearson como en Spearman, lo que sugiere que la mayoría de las relaciones son monótonas. Sin embargo, en la matriz de Spearman se detectan algunas relaciones que no se habían visto en Pearson, lo que indica que existen patrones no lineales en los datos que la correlación lineal no logra captar.

# ## Tratamiento de valores nulos

# Antes de decidir cómo manejar los valores nulos, es fundamental analizar su distribución en relación con la variable objetivo. Específicamente, es útil determinar si los valores faltantes se concentran en una clase específica de la variable objetivo o si su distribución es uniforme.

# In[30]:


list_var_continuous


# In[31]:


def get_percent_null_values_target(pd_loan, list_var_continuous, target):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función get_percent_null_values_target:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función analiza la relación entre los valores nulos de variables continuas 
        y la variable objetivo. Identifica si los valores faltantes de cada variable se 
        distribuyen de forma uniforme respecto a las clases de la variable objetivo o si 
        están asociados de manera significativa a alguna de ellas.
        
    - Inputs: 
        - pd_loan (DataFrame): DataFrame que contiene los datos de entrada.
        - list_var_continuous (list): Lista de nombres de variables continuas a analizar.
        - target (str): Nombre de la variable objetivo.

    - Output:
        - pd_final (DataFrame): DataFrame que resumen del analisis de la relación entre 
          los valores nulos de variables continuas y la variable objetivo.
    ----------------------------------------------------------------------------------------------------------
    '''
    
    # DataFrame final donde se acumularán los resultados
    pd_final = pd.DataFrame()
    
    # Iterar sobre cada variable continua de la lista
    for i in list_var_continuous:
        if pd_loan[i].isnull().sum()>0:
            target_distribution = pd_loan[target][pd_loan[i].isnull()].value_counts(normalize=True)
            
            target_dict = target_distribution.to_dict()
            percent_0 = target_dict.get(0, 0)  
            percent_1 = target_dict.get(1, 0)  

            # Crear un DataFrame temporal con la estructura deseada
            temp_df = pd.DataFrame({
                '0': [percent_0],
                '1': [percent_1],
                'variable': [i],
                'sum_null_values': [pd_loan[i].isnull().sum()],
                'porcentaje_sum_null_values': [pd_loan[i].isnull().sum() / pd_loan.shape[0]]
            })
            pd_final = pd.concat([pd_final, temp_df], axis=0).reset_index(drop=True)
    
    # Si no se encuentran variables con valores nulos, mostrar un mensaje
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final


# In[32]:


get_percent_null_values_target(pd_loan_train, list_var_continuous, target='TARGET')


# Como se comentó en el notebook anterior, el muestreo está desbalanceado, con la mayoría de los clientes sin dificultades para pagar el préstamo. Además, los valores faltantes se concentran principalmente en la clase 0 de la variable objetivo (clientes sin problemas de pago), siguiendo la misma tónica del desbalance presente en los datos.

# Dado que al principio no cuento con suficiente contexto sobre las variables, se pueden emplear diferentes enfoques y comparar los resultados del modelo. Las opciones son las siguientes:
# 
# - Opción 0: Algunos algoritmos pueden manejar valores faltantes directamente, sin necesidad de imputarlos.
# 
# - Opción 1: Eliminar filas con valores nulos. No obstante, esta opción no es ideal en mi caso, ya que, como se ha observado, hay una cantidad significativa de filas con datos faltantes.
# 
# - Opción 2: Imputar los valores faltantes mediante técnicas estadísticas como la media, mediana, máximo, mínimo o incluso valores extremos.
# 
# - Opción 3: Rellenar los valores faltantes utilizando modelos de regresión, como KNN, regresión lineal o XGBoost. Sin embargo, este enfoque podría implicar un alto costo computacional y requiere cuidado para evitar el sobreajuste.

# 
# En este trabajo he optado por la opción 2 para imputar los valores faltantes utilizando la media o mediana. Esto es ideal para completar los datos sin recurrir a métodos costosos ni distorsionar su distribución. Descarto el uso de valores extremos, ya que las hay columnas que representan diferencias de tiempo, con valores tanto negativos como positivos. Además, Utilizar extremos podría introducir sesgos y afectar la coherencia del análisis.
# 
# Por lo tanto, imputar los valores faltantes con la media o la mediana permite mantener la coherencia y el equilibrio de las columnas, asegurando que los valores reemplazados sean representativos de la tendencia general de los datos sin introducir sesgos indebidos.
# 
# Es importante recordar que los valores faltantes en el conjunto de test se imputan utilizando la media calculada a partir del conjunto de train.

# In[33]:


pd_loan_train[list_var_continuous] = pd_loan_train[list_var_continuous].apply(lambda x: x.fillna(x.median()))
pd_loan_test[list_var_continuous] = pd_loan_test[list_var_continuous].apply(lambda x: x.fillna(x.median()))


# In[34]:


pd_loan_train[list_var_continuous]


# In[35]:


pd_loan_train.shape


# Ahora verificamos que no hay variables con valores nulos, confirmando que el reemplazo se ha realizado con éxito.

# In[36]:


get_percent_null_values_target(pd_loan_train, list_var_continuous, target='TARGET')


# In[37]:


list_var_continuous = list(pd_loan_train.select_dtypes('float').columns)
get_corr_matrix(dataset = pd_loan_train[list_var_continuous], 
                metodo='pearson', size_figure=[10,8])


# Dado que se han imputado los valores faltantes utilizando la mediana, al revisar la matriz de correlación de Pearson no se han observado diferencias significativas en comparación con los datos originales. Esto significa que el método de imputación ha funcionado bien y no ha alterado las relaciones entre las variables en el conjunto de datos.

# ## Tratamiento de las variables categoricas

# 
# El tratamiento de las variables categóricas incluye evaluar su relación y relevancia para el modelo. La medida V-Cramér evalúa la fuerza de la asociación entre dos variables categóricas, con valores cercanos a 1 indicando una fuerte relación. Este análisis ayuda a seleccionar variables relevantes y eliminar redundancias.

# In[38]:


def cramers_v(confusion_matrix):
    ''' 
    ----------------------------------------------------------------------------------------------------------
    Función cramers_v:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función calcula el estadístico V de Cramér para medir la asociación entre dos 
        variables categóricas. Utiliza la corrección de Bergsma y Wicher (2013) para ajustar
        el valor del chi-cuadrado y calcular una medida que indique la fuerza de la relación 
        entre las variables. El valor de Cramér's V oscila entre 0 (sin asociación) y 1 
        (asociación perfecta).
        
    - Inputs: 
        - confusion_matrix (DataFrame): Tabla de contingencia que contiene las frecuencias 
        absolutas de las categorías de las dos variables a analizar.
        
    - Output:
        - float: Valor de la V de Cramér 
    ----------------------------------------------------------------------------------------------------------    
    '''
    
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


# In[39]:


confusion_matrix = pd.crosstab(pd_loan_train["TARGET"], pd_loan_train["TARGET"])
cramers_v(confusion_matrix.values)


# Analizamos la medida V de Cramér para cada una de las variables categóricas.

# In[40]:


# Lista de variables categóricas excluyendo 'TARGET'
list_var_cat = [var for var in list_var_cat if var != "TARGET"]

# Iterar sobre las variables categóricas
for var in list_var_cat:
    print(f"Variable: {var}")
    
    # Matriz de confusión
    confusion_matrix = pd.crosstab(pd_loan_train["TARGET"], pd_loan_train[var])
    print("Confusion Matrix:")
    print(confusion_matrix)
    
    # Calcular Cramér's V
    cramer_v_value = cramers_v(confusion_matrix.values)
    print(f"Cramér's V: {cramer_v_value}")
    print("-" * 50)


# Las variables analizadas muestran asociaciones débiles con la variable objetivo (`TARGET`), según los valores de Cramér's V, la mayoría menores a 0.1, indicando baja relación entre ellas y el incumplimiento de pagos. Esto sugiere que otras variables o enfoques podrían ser necesarios para mejorar el modelo predictivo.

# Adicionalmente al coeficiente V de Cramer, he analizado en las variables categóricas el Weight of Evidence (WOE) y el Information Value (IV). El WOE mide la fuerza de la relación entre una variable categórica y la variable objetivo, transformando las categorías en valores continuos que reflejan el riesgo de un evento. Por su parte, el IV cuantifica la capacidad predictiva de una variable categórica, indicando qué tan bien una variable distingue entre las clases del objetivo.

# In[41]:


def calculate_woe(df, target, feature):
    '''
    Función para calcular el WOE de una variable categórica en relación a un target binario.
    
    Parámetros:
    - df (DataFrame): DataFrame de pandas que contiene los datos.
    - target (str): Nombre de la variable objetivo.
    - feature (str): Nombre de la columna de la variable categórica.
    
    Retorna:
    - WOE (DataFrame): DataFrame con las categorías de la variable y su WOE correspondiente.
    '''
    
    # Crear tabla de contingencia entre feature y target
    cross_tab = pd.crosstab(df[feature], df[target])
    
    # Calcular el número total de eventos y no-eventos
    total_events = cross_tab.sum(axis=0)[1]  # Suma de eventos (1)
    total_non_events = cross_tab.sum(axis=0)[0]  # Suma de no-eventos (0)
    
    # Calcular las proporciones de eventos (1) y no-eventos (0) por categoría
    cross_tab['p_event'] = cross_tab[1] / total_events
    cross_tab['p_non_event'] = cross_tab[0] / total_non_events
    
    # Calcular WOE para cada categoría
    cross_tab['WOE'] = np.log(cross_tab['p_non_event'] / cross_tab['p_event'])
    
    # Filtrar solo las categorías y su WOE
    woe_values = cross_tab[['WOE']]
    
    return woe_values


# In[44]:


# Iterar sobre las variables categóricas
for var in list_var_cat:
    print(f"Variable: {var}")
    # Calcular WOE
    woe_value = calculate_woe(pd_loan_train, 'TARGET', var)
    print(f"IV: {woe_value}")
    print("-" * 50)


# Teniendo en cuenta que la variable objetivo indica si el cliente ha tenido dificultades de pago (1) o no (0), los valores de WOE reflejan cómo las categorías de las variables afectan la probabilidad de que un cliente tenga problemas de pago.
# 
# Por ejemplo, en la variable de nivel educativo del cliente (`NAME_EDUCATION_TYPE`), el valor de WOE para los que tienen título académico (1.33) sugiere que estos clientes tienen una mayor probabilidad de no tener dificultades de pago. En cambio, los clientes con secundaria incompleta tienen un valor de WOE negativo (-0.30), lo que indica que tienen más probabilidades de experimentar dificultades de pago.
# 
# En cuanto al tipo de contrato (`NAME_CONTRACT_TYPE`), los préstamos revolventes tienen un WOE positivo (0.41), lo que sugiere que los clientes con este tipo de préstamo tienen menos probabilidades de tener problemas de pago. Por otro lado, los préstamos en efectivo tienen un WOE cercano a cero, lo que implica que hay una relación débil con la probabilidad de impago.
# 
# Como mencionamos antes, la variable de tipo de ingreso (`NAME_INCOME_TYPE`) es clave. Los empresarios y estudiantes tienen un WOE infinito, lo que indica que tienen una relación muy fuerte con un bajo riesgo de impago. En cambio, los desempleados tienen un WOE negativo (-1.81), lo que refleja un alto riesgo de dificultades de pago.
# 
# También es relevante el WOE de la cantidad de hijos (`CNT_CHILDREN`). A medida que el número de hijos aumenta, el WOE disminuye, lo que sugiere que los clientes con más hijos tienen una mayor probabilidad de enfrentar dificultades de pago.
# 
# Por último, en cuanto al tipo de ocupación (`OCCUPATION_TYPE`), las ocupaciones de contables y gerentes tienen valores de WOE positivos, lo que indica un menor riesgo de impago, mientras que los trabajadores no calificados tienen un WOE negativo significativo (-0.78), lo que sugiere una mayor probabilidad de dificultades de pago.

# In[45]:


def calculate_iv(df, target, feature):
    '''
    Función para calcular el Information Value (IV) de una variable categórica en relación al target binario.
    
    Parámetros:
    - df (DataFrame): DataFrame de pandas que contiene los datos.
    - target (str): Nombre de la columna objetivo binaria.
    - feature (str): Nombre de la columna de la variable categórica.
    
    Retorna:
    - IV (float): El valor del Information Value.
    '''
    
    # Crear tabla de contingencia entre feature y target
    cross_tab = pd.crosstab(df[feature], df[target])
    
    # Calcular el número total de eventos y no-eventos
    total_events = cross_tab.sum(axis=0)[1]  # Suma de eventos (1)
    total_non_events = cross_tab.sum(axis=0)[0]  # Suma de no-eventos (0)
    
    # Calcular las proporciones de eventos (1) y no-eventos (0) por categoría
    cross_tab['p_event'] = cross_tab[1] / total_events
    cross_tab['p_non_event'] = cross_tab[0] / total_non_events
    
    # Calcular WOE para cada categoría
    cross_tab['WOE'] = np.log(cross_tab['p_non_event'] / cross_tab['p_event'])
    
    # Calcular IV sumando el producto de la diferencia de proporciones y WOE
    cross_tab['IV'] = (cross_tab['p_non_event'] - cross_tab['p_event']) * cross_tab['WOE']
    
    # Calcular el IV total
    iv_value = cross_tab['IV'].sum()
    
    return iv_value


# In[46]:


# Iterar sobre las variables categóricas
for var in list_var_cat:
    print(f"Variable: {var}")
    # Calcular IV
    iv_value = calculate_iv(pd_loan_train, 'TARGET', var)
    print(f"IV: {iv_value}")
    print("-" * 50)


# En resumen, las variables con un IV alto, como `OCCUPATION_TYPE` y `NAME_EDUCATION_TYPE`, tienen un mayor poder predictivo y son útiles para predecir las dificultades de pago. Por otro lado, las variables con un IV bajo o cercano a cero, como `FLAG_MOBIL` y `FLAG_CONT_MOBILE`, no aportan mucha información y podrían no ser relevantes para el modelo. En general, las variables con un IV más alto son las más importantes para el modelo, mientras que aquellas con IV bajo o infinito son candidatas a ser revisadas para evaluar su utilidad.

# ## Tratamiento de valores nulos

# En las variables categóricas, los valores nulos suelen reemplazarse asignando una nueva categoría: "Sin valor". 

# In[47]:


pd_loan_train[list_var_cat] = pd_loan_train[list_var_cat].astype("object").fillna("SIN VALOR").astype("category")
pd_loan_test[list_var_cat] = pd_loan_test[list_var_cat].astype("object").fillna("SIN VALOR").astype("category")


# Es importante recordar que los datos de tipo entero definidos como categóricos, incluidas las variables booleanas, no presentan valores nulos. Sin embargo, si los tuvieran, deberían haberse tratado como parte del manejo de valores nulos numéricos. Para las variables booleanas, una opción sería imputar los nulos con -1, mientras que para las demás variables categóricas numéricas, sería necesario analizarlas con más detalle. Siguiendo el enfoque empleado hasta ahora, podríamos imputar los nulos con la mediana.

# ## Guardado de la tabla

# Ahora, tras un segundo procesamiento y análisis de los datos, se guarda el DataFrame para conservar este nuevo estado intermedio y facilitar su uso en futuras etapas del análisis.

# In[48]:


pd_loan_train.to_csv("../data/interim/train_pd_data_preprocessing_missing_outlier.csv")
pd_loan_test.to_csv("../data/interim/test_pd_data_preprocessing_missing_outlier.csv")


# In[49]:


print(pd_loan_train.shape, pd_loan_test.shape)

