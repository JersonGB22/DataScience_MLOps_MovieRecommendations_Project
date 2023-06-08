<p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

# <h1 align=center> **PROYECTO INDIVIDUAL N°1: Machine Learning Operations (MLOps)** </h1>

<p align="center">
  <img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png" height="250" width="auto" style="max-width: 50%;" />
  <img src="Datasets/imagen_imdb.jpg" height="250" width="auto" style="max-width: 50%;" />
</p>
<center>

*Bienvenid@s a la presentación de mi primer proyecto de la carrera de Data Science, el cual fue realizado durante la etapa de Labs del bootcamp [Henry](https://www.soyhenry.com/carrera-data-science)*

</center>

# **Descripción**:

Este proyecto individual tiene como objetivo simular el papel de un `Data Scientist` en una startup que se dedica a la agregación de plataformas de streaming. El objetivo es desarrollar consultas de películas y un sistema de recomendación basado en `Machine Learning` que, hasta ahora, no ha podido ser implementado debido a la falta de madurez en los datos. Para llevar a cabo este proyecto, se necesitará realizar tareas de `Data Engineer` para la recolección y tratamiento de los datos, así como el entrenamiento del modelo de Machine Learning y finalmente, implementarlo en el mundo real con la metodología de `DevOps`.

***Se recomienda encarecidamente al lector revisar los notebooks y el script asociados en los enlaces proporcionados en las etapas del proyecto. Estos archivos contienen explicaciones exhaustivas y comentarios detallados que describen cada paso de la realización del proyecto.***

# **Etapa I: Transformaciones**
**Consignas:**  

+ Algunos campos, como **`belongs_to_collection`**, **`production_companies`** y otros (ver diccionario de datos) están anidados, esto es o bien tienen un diccionario o una lista como valores en cada fila, ¡deberán desanidarlos para poder  y unirlos al dataset de nuevo hacer alguna de las consultas de la API! O bien buscar la manera de acceder a esos datos sin desanidarlos.

+ Los valores nulos de los campos **`revenue`**, **`budget`** deben ser rellenados por el número **`0`**.
  
+ Los valores nulos del campo **`release date`** deben eliminarse.

+ De haber fechas, deberán tener el formato **`AAAA-mm-dd`**, además deberán crear la columna **`release_year`** donde extraerán el año de la fecha de estreno.

+ Crear la columna con el retorno de inversión, llamada **`return`** con los campos **`revenue`** y **`budget`**, dividiendo estas dos últimas **`revenue / budget`**, cuando no hay datos disponibles para calcularlo, deberá tomar el valor **`0`**.

+ Eliminar las columnas que no serán utilizadas, **`video`**,**`imdb_id`**,**`adult`**,**`original_title`**,**`poster_path`** y **`homepage`**.

**Resolución de la consignas:**

Las transformaciones requeridas por las [especificaciones del proyecto](https://github.com/JersonGB22/ProyectoIndividual1_MLOps_Henry/blob/main/Consignas/Readme.md) corresponden a la fase de ETL (Extract, Transform and Load), que se llevó a cabo en un archivo Jupyter Notebook llamado [transformaciones.ipynb](https://github.com/JersonGB22/ProyectoIndividual1_MLOps_Henry/blob/main/Transformaciones.ipynb), utilizando los archivos CSVs ``movies_dataset.csv`` y `credits.csv` presentes en la carpeta [Datasets](https://github.com/JersonGB22/ProyectoIndividual1_MLOps_Henry/tree/main/Datasets). Se emplearon las librerías Pandas y NumPy. Finalmente, el dataframe resultante se exportó en el archivo CSV [movie_transformation.csv](https://raw.githubusercontent.com/JersonGB22/ProyectoIndividual1_MLOps_Henry/main/Datasets/movie_transformation.csv) alojado en la carpeta Datasets para ser utilizado en la etapa de implementación de los endpoints que se consumirán en la API.

# **Etapa II: Desarrollo API**
**Consignas:** 
Creas 6 funciones para disponibilizar los datos de la empresa usando el framework ***FastAPI***
+ def **cantidad_filmaciones_mes( *`Mes`* )**:
    Se ingresa un mes en idioma Español. Debe devolver la cantidad de películas que fueron estrenadas en el mes consultado en la totalidad del dataset.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ejemplo de retorno: *`X` cantidad de películas fueron estrenadas en el mes de `X`*

+ def **cantidad_filmaciones_dia( *`Dia`* )**:
    Se ingresa un día en idioma Español. Debe devolver la cantidad de películas que fueron estrenadas en día consultado en la totalidad del dataset.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ejemplo de retorno: *`X` cantidad de películas fueron estrenadas en los días `X`*

+ def **score_titulo( *`titulo_de_la_filmación`* )**:
    Se ingresa el título de una filmación esperando como respuesta el título, el año de estreno y el score.
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ejemplo de retorno: *La película `X` fue estrenada en el año `X` con un score/popularidad de `X`*

+ def **votos_titulo( *`titulo_de_la_filmación`* )**:
    Se ingresa el título de una filmación esperando como respuesta el título, la cantidad de votos y el valor promedio de las votaciones. La misma variable deberá de contar con al menos 2000 valoraciones, caso contrario, debemos contar con un mensaje avisando que no cumple esta condición y que por ende, no se devuelve ningun valor.
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ejemplo de retorno: *La película `X` fue estrenada en el año `X`. La misma cuenta con un total de `X` valoraciones, con un promedio de `X`*

+ def **get_actor( *`nombre_actor`* )**:
    Se ingresa el nombre de un actor que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. Además, la cantidad de películas que en las que ha participado y el promedio de retorno. **La definición no deberá considerar directores.**
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ejemplo de retorno: *El actor `X` ha participado de `X` cantidad de filmaciones, el mismo ha conseguido un retorno de `X` con un promedio de `X` por filmación*

+ def **get_director( *`nombre_director`* )**:
    Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.
**Resolución de la consignas:**

En esta etapa del proyecto, hemos creado los seis endpoints que se nos han asignado mediante el uso de las librerías pandas y FastAPI en el archivo [main.py](https://github.com/JersonGB22/ProyectoIndividual1_MLOps_Henry/blob/main/main.py). Además, hemos incluido algunas observaciones importantes que es necesario tomar en cuenta. En cada función, se han considerado los casos en los que los argumentos no están presentes en el dataframe establecido, y se han proporcionado instrucciones sobre qué argumentos deben ser ingresados. 

# **Etapa III: Análisis Exploratorio de los Datos (EDA)**

**Consigna:**
Ya los datos están limpios, ahora es tiempo de investigar las relaciones que hay entre las variables de los datasets, ver si hay outliers o anomalías (que no tienen que ser errores necesariamente :eyes: ), y ver si hay algún patrón interesante que valga la pena explorar en un análisis posterior.  Sabes que puedes apoyarte en librerías como _pandas profiling, sweetviz, autoviz_, entre otros y sacar de allí tus conclusiones

**Resolución de la consigna:**

Esta etapa se realizó en el archivo [EDA_ETL(ML)](https://github.com/JersonGB22/ProyectoIndividual1_MLOps_Henry/blob/main/EDA_ETL(ML).ipynb). En base al archivo `movie_transformation`, en esta etapa se realizó un [EDA inicial general](https://huknt1mctoiumyo41mctaq.on.drv.tw/ArchivosHTML/ReportEDA_MovieData_IMDb.html) utilizando la librería ydata-profiling (antes llamada pandas-profiling). Posteriormente, se llevó a cabo un EDA más detallado. Durante este proceso, se identificaron outliers, se examinaron las distribuciones y correlaciones de las variables numéricas, y se revisaron las frecuencias de las variables categóricas. Estas tareas se realizaron utilizando las librerías matplotlib y seaborn. Después, se procedió a realizar tanto un ETL como un EDA paralelos, los cuales eran necesarios para el desarrollo del sistema de recomendación basado en contenido. En el proceso de ETL se eliminaron columnas innecesarias, se imputaron valores faltantes y se realizó el preprocesamiento del texto. Cabe destacar que el preprocesamiento del texto es una tarea de procesamiento del lenguaje natural (NLP), y para llevarla a cabo se utilizaron las librerías regex, stop_words y nltk. En cuanto al EDA, se generaron gráficos de barras (barplots) para visualizar las características más relevantes de las películas. Además, se crearon nubes de palabras (word cloud) utilizando las variables ``title`` y ``overview``, gracias a las librerías stylecloud y googletrans. Finalmente, se exportó el dataframe generado en un archivo CSV denominado [ML_movie_transformation.csv](https://raw.githubusercontent.com/JersonGB22/ProyectoIndividual1_MLOps_Henry/main/Datasets/ML_movie_transformation.csv), el cual se utilizará en la etapa de Machine Learning.


# **Etapa IV: Machine Learning (ML)**
 **Consigna:**
 Realizar un modelo de machine learning para armar un sistema de recomendación de películas. Éste consiste en recomendar películas a los usuarios basándose en películas similares, por lo que se debe encontrar la similitud de puntuación entre esa película y el resto de películas, se ordenarán según el score y devolverá una lista de Python con 5 valores, cada uno siendo el string del nombre de las películas con mayor puntaje, en orden descendente. Debe ser deployado como una función adicional de la API anterior y debe llamarse 
 + def **recomendacion( *`titulo`* )**:
    Se ingresa el nombre de una película y te recomienda las similares en una lista de 5 valores.

**Resolución de la consigna:**

La etapa de construcción del modelo de machine learning, se llevó a cabo en el archivo [MachineLearning.ipynb](https://github.com/JersonGB22/ProyectoIndividual1_MLOps_Henry/blob/main/MachineLearning.ipynb) utilizando la biblioteca scikit-learn (sklearn). Primero, cargamos el conjunto de datos `ML_movie_transformation.csv`, y luego creamos una matriz de características utilizando la clase `CountVectorizer`. A continuación, entrenamos y transformamos el modelo, lo que resultó en un vector. Luego, utilizamos la clase `cosine_similarity` para calcular la matriz de similitud. Además, construimos una función para obtener una lista de índices de las películas con mayor puntaje en orden descendente en relación a una película específica. Esta función se aplicó a la columna `label` para obtener las listas correspondientes a cada película, lo que nos permitió hacer recomendaciones personalizadas de manera más rápida y efectiva para el endpoint 7 de la API.

Finalmente, exportamos el dataframe resultante a un archivo CSV denominado [API_ML_movie.csv](https://raw.githubusercontent.com/JersonGB22/ProyectoIndividual1_MLOps_Henry/main/Datasets/API_ML_movie.csv), el cual se utilizó para realizar la función encomendada `recomendacion`, en el archivo `main.py`, la cual recibe el nombre de una película y devuelve las 5 más parecidas a esta en orden descendente. También valida el argumento `titulo` y proporciona instrucciones sobre los argumentos necesarios para su uso, al igual que las otras seis funciones.

# **Etapa V: Deployment**
 **Consigna:**
Conoces sobre Render y tienes un tutorial de Render que te hace la vida mas facil. También podrías usar Railway, o cualquier otro servicio que permita que la API pueda ser consumida desde la web.

 **Resolución de la consigna:**

Para hacer el deployment de nuestro proyecto, se utilizó Render, un servicio de alojamiento en la nube para aplicaciones web. En nuestro archivo `main.py`, se encuentran los siete endpoints que hemos desarrollado utilizando la biblioteca FastAPI y Pandas. Cada endpoint tiene una función asociada que realiza la tarea correspondiente, como la recuperación de datos de un conjunto de datos, el procesamiento de datos y la generación de recomendaciones personalizadas. Antes de hacer el deployment en Render, es importante crear un archivo [requirements.txt](https://github.com/JersonGB22/ProyectoIndividual1_MLOps_Henry/blob/main/requirements.txt) que especifique las bibliotecas y versiones utilizadas en nuestro proyecto. Esto asegura que Render pueda instalar las mismas bibliotecas en su servidor, lo que garantiza la compatibilidad de versiones y evita problemas de incompatibilidad que puedan surgir durante el deployment.

Una vez que hemos creado el archivo `requirements.txt`, podemos seguir las instrucciones de Render para hacer el deployment de nuestro proyecto. Render nos proporciona una URL para acceder a nuestra aplicación, lo que nos permite probar y verificar que todo funciona como se espera.
## Pasos para hacer el deployment de nuestro proyecto en Render:
- Iniciar un repositorio Git en la carpeta de tu proyecto local, utilizando el comando: `git init`.
- Crear un nuevo repositorio en GitHub para el proyecto.
- Vincular tu repositorio local de Git con tu repositorio en GitHub, con el siguiente comando: `git remote add origin <URL de tu repositorio en GitHub>`
- Realizar un commit de los cambios en tu proyecto y un push de los cambios a tu repositorio en GitHub, con los siguientes comandos respectivamente: `git add`, `git commit -m "Descripción del commit"` y `git push -u origin master`. Para archivos que superan los 100 Mb, como en el caso específico del archivo `credits.csv`, se optó por utilizar Git LFS. Esto se logró ejecutando el siguiente comando: `git lfs track "Datasets/credits.csv"`.
- Crear una nueva aplicación en Render. Iniciar sesión en Render y hacer clic en el botón "Add a New Service". Seleccionar el tipo de servicio "Web Service" y seguir las instrucciones para configurar la aplicación.
- Vincular la aplicación de Render con el repositorio de GitHub. En la sección de configuración de la aplicación en Render, haz clic en "Link to GitHub" y selecciona el repositorio que acabas de crear.
- Hacer un deploy de la aplicación en Render. En la sección de configuración de la aplicación en Render, hacer clic en el botón "Deploy". Render se encargará de clonar el repositorio de GitHub y desplegar tu aplicación en línea.
- Verificar que la aplicación está funcionando correctamente. 
## [URL de la aplicación web](https://jersonbgb-projectoindividual1-imdb-henry.onrender.com/)

- Nota: Para acceder a la documentación de la API, visite la siguiente URL en su navegador: <https://jersonbgb-projectoindividual1-imdb-henry.onrender.com/docs>. Al hacerlo, se abrirá la página de documentación interactiva, la cual le permitirá explorar todos los endpoints disponibles y probarlos en tiempo real.
La documentación interactiva ha sido generada automáticamente por FastAPI, y en ella encontrará una descripción detallada de cada uno de los endpoints, así como de los parámetros que aceptan y los esquemas de respuesta. De esta manera, podrá tener un mayor entendimiento sobre el funcionamiento de la API y cómo utilizarla de manera efectiva.

## [Link al video explicativo del proyecto](https://drive.google.com/file/d/1Dlg6_kQGin3Lw1qxQvw_eMI7tyDLgMKP/view?usp=sharing)

# **Tecnologías utilizadas:**
| Technology | Documentation |
|------------|---------------|
| Visual Studio Code (VSC) | https://code.visualstudio.com/docs |
| Python | https://docs.python.org/3/ |
| Render | https://render.com/docs |
| Git | https://git-scm.com/doc |
| GitHub | https://docs.github.com/en |
| Git LFS | https://git-lfs.com/ |
| Pandas | https://pandas.pydata.org/docs/ |
| Scikit-learn (sklearn) | https://scikit-learn.org/stable/documentation.html |
| FastAPI | https://fastapi.tiangolo.com/ |
| Uvicorn | https://www.uvicorn.org/ |
| NumPy | https://numpy.org/doc/ |
| Ydata-Profiling | https://ydata-profiling.ydata.ai/docs/master/index.html |
| Matplotlib | https://matplotlib.org/stable/contents.html |
| Seaborn | https://seaborn.pydata.org/ |
| NLTK | https://www.nltk.org/ |
| Regex (expresiones regulares) | https://docs.python.org/3/library/re.html |
| Stop words | https://pypi.org/project/stop-words/ |
| Stylecloud | https://pypi.org/project/stylecloud/ |
| Googletrans | https://pypi.org/project/googletrans/ |


<p>
  <img src="https://code.visualstudio.com/assets/favicon.ico" height="100" width="auto" style="max-width: 50%;" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" height="100" width="auto" style="max-width: 50%;" />
  <img src="https://images.g2crowd.com/uploads/product/image/large_detail/large_detail_477db83f729d63210139ec7cd29c1351/render-render.png" height="100" width="auto" style="max-width: 50%;" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Git_icon.svg/1200px-Git_icon.svg.png" height="100" width="auto" style="max-width: 50%;" />
  <img src="https://github.com/fluidicon.png" height="100" width="auto" style="max-width: 50%;" />
  <img src="https://img2.helpnetsecurity.com/posts2020/git-lfs-logo.jpg" height="100" width="auto" style="max-width: 50%;" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/2560px-Pandas_logo.svg.png" height="100" width="auto" style="max-width: 50%;" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/640px-Scikit_learn_logo_small.svg.png" height="100" width="auto" style="max-width: 50%;" />
  <img src="https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png" height="100" width="auto" style="max-width: 50%;" />
  <img src="https://raw.githubusercontent.com/tomchristie/uvicorn/master/docs/uvicorn.png" height="100" width="auto" style="max-width: 50%;" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/1200px-NumPy_logo_2020.svg.png" height="100" width="auto" style="max-width: 50%;" />
  <img src="https://camo.githubusercontent.com/350907f8f83a7a83e678cfdd24b85ecc3948a2cc469a86c61d2db7e393246626/68747470733a2f2f6173736574732e79646174612e61692f6f73732f79646174612d70726f66696c696e675f626c61636b2e706e67" height="100" width="auto" style="max-width: 50%;" />
  <img src="https://d33wubrfki0l68.cloudfront.net/e33fd6f372aa5d51e7b0de4bd763bd983251881e/4b0f4/blog/customising-matplotlib/matplot_title_logo.png" height="100" width="auto" style="max-width: 50%;" />
  <img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" height="80" width="auto" style="max-width: 50%;" />
  <img src="https://miro.medium.com/v2/resize:fit:592/0*zKRz1UgqpOZ4bvuA" height="80" width="auto" style="max-width: 50%;" />
  <img src="https://www.freecodecamp.org/news/content/images/2023/02/regexpy.png" height="80" width="auto" style="max-width: 50%;" />
</p>

# **Datos del Autor:**
## ***Jerson Brayan Gimenes Beltrán***

### **Correo electrónico:** jerson.gimenesbeltran@gmail.com
