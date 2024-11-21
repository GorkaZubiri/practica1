# Práctica 1: Análisis Exploratorio de Datos

**Autor:** Gorka Zubiri Elso

**Correo electrónico:** gorka.zubiri@cunef.edu

**Directorio GitHub:** https://github.com/GorkaZubiri/practica1

Esta práctica tiene como objetivo ayudar a un banco a mejorar la aprobación de préstamos mediante el análisis de datos históricos de solicitudes. La idea es identificar patrones en los clientes que cumplen con sus pagos y aquellos que no lo hacen, para aplicar esta información en futuros modelos predictivos.

## Objetivos principales de este trabajo

1. **Explorar y entender el dataset**: Familiarizarme con las variables que contiene el conjunto de datos.
2. **Análisis de las variables**: Realizar un análisis general de las variables y su distribución, con especial atención a su relación con la variable objetivo (dificultad de pago).
3. **Depuración de los datos**: Limpiar y preparar el dataset para la aplicación de modelos predictivos, gestionando adecuadamente los valores nulos, outliers y tipos de datos erróneos.
4. **Preprocesamiento para Machine Learning**: Codificar las variables categóricas y escalar las variables numéricas para que los datos estén listos para futuros modelos.
5. **Análisis descriptivo**: Extraer conclusiones del análisis exploratorio que permitan entender los componentes clave del dataset y su relación con la variable objetivo.

## Análisis Exploratorio de Datos (EDA)

Este análisis tiene como propósito entender las principales características del conjunto de datos y cómo estas se relacionan con la probabilidad de incumplimiento de pago. Los resultados obtenidos en este análisis son los siguientes:

### 1. Comprensión de las variables

A través de un análisis preliminar, se ha logrado entender el significado y la distribución de cada una de las variables presentes en el dataset.

### 2. Estudio de la distribución y naturaleza de los datos

Se realizó un análisis detallado de la distribución de las variables tanto numéricas como categóricas. Además, se utilizaron visualizaciones gráficas (como histogramas, diagramas de dispersión y boxplots) para observar la distribución de las variables y sus relaciones con la variable objetivo.

### 3. División del dataset en train y test

Para garantizar una correcta evaluación de los modelos, se realizó una división estratificada del dataset en dos subconjuntos: entrenamiento (train) y prueba (test).

### 4. Identificación de valores nulos y outliers

Durante la exploración de los datos, se identificaron variables con valores faltantes y outliers. Se tomaron decisiones adecuadas para el tratamiento de estos problemas, considerando la naturaleza de cada variable y su posible impacto en el análisis.

### 5. Correlaciones entre variables

Se analizaron las correlaciones entre las variables numéricas y categóricas utilizando herramientas estadísticas como el coeficiente de Pearson para las variables continuas, y el coeficiente V de Cramer y Weight of Evidence (WOE) para las variables categóricas, entre otros.

### 6. Codificación de variables categóricas

Para transformar las variables categóricas en un formato numérico adecuado para modelos de machine learning, se aplicaron técnicas como One-Hot Encoding y Target Encoding.

### 7. Escalado de variables numéricas

Finalmente, se aplicaron técnicas de escalado a las variables numéricas para normalizar su rango y asegurar que todas las características tengan la misma importancia en el análisis.

## Estructura del Directorio

La estructura del directorio de este proyecto está organizada de la siguiente manera:

- **`data/`**: Contiene los archivos de datos con los que vamos a trabajar.

  - **`raw/`**: Archivos de datos originales, tal como se obtuvieron.
  
  - **`processed/`**: Datos que ya han sido procesados y transformados para su uso.
  
  - **`interim/`**: Datos intermedios que han sido parcialmente procesados y aún no están listos para su uso final.
  
  
- **`env/`**: Archivos relacionados con el entorno de desarrollo, incluyendo un archivo `requirements.txt` con todas las librerías y dependencias utilizadas en el proyecto.


- **`notebooks/`**: Contiene los notebooks en formato Jupyter (`.ipynb`) que documentan el análisis de datos y otros experimentos.


- **`html/`**: Carpeta donde se almacenan los notebooks convertidos en formato HTML para facilitar su visualización y compartición.


- **`src/`**: Directorio que guarda los archivos fuente de Python, tales como scripts, funciones o clases utilizadas en el procesamiento de datos o la creación de modelos.

## Notebooks Desarrollados

Se han desarrollado tres notebooks en esta primera práctica:

1. **00_Exploracion_General**: Su objetivo es comprender las variables y estudiar la distribución y naturaleza de los datos.
2. **01_EDA_Procesamiento_Vars**: Enfocado en la división del dataset en train y test, la identificación de valores nulos y outliers, y el análisis de correlaciones entre variables.
3. **02_Codificacion_Var_Escalado**: Se centra en la codificación de variables categóricas y el escalado de variables numéricas.