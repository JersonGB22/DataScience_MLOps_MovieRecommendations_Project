# Importación de las librerías necesarias
import pandas as pd
from fastapi import FastAPI

# Importación de los datos
df_movie=pd.read_csv(r"https://raw.githubusercontent.com/JersonGB22/ProyectoIndividual1_MLOps_Henry/main/Datasets/movie_transformation.csv")
# Conversión de la columna 'release_date' a tipo datetime
df_movie["release_date"]=pd.to_datetime(df_movie.release_date)
# Conversión de los campos necesarios a tipo list
df_movie["actor"]=df_movie.actor.apply(lambda x: eval(x))
df_movie["director"]=df_movie.director.apply(lambda x: eval(x))
# Conversión de los campos necesarios a minúsculas para optimizar las consultas
df_movie["actor"]=df_movie.actor.apply(lambda x: [i.lower() for i in x])
df_movie["director"]=df_movie.director.apply(lambda x: [i.lower() for i in x])
df_movie["title"]=df_movie.title.apply(lambda x: x.lower())

# Data frame para el endpoint de ML
df_ml=pd.read_csv(r"https://raw.githubusercontent.com/JersonGB22/ProyectoIndividual1_MLOps_Henry/main/Datasets/API_ML_movie.csv")

# Instanciamos la clase FastAPI para construir la aplicación de Interfaz de Consultas
app=FastAPI()

# Creación del endpoint de Bienvenida
@app.get("/")
def welcome():
    return "Bienvenid@s a mi Proyecto Individual Nº1: Machine Learning Operations (MLOps)"

## Creación de los endpoints

"""
OBSERVACIÓN: Por motivos de que Render no acepta la configuración regional que admite el idioma español (es) se tiene que crear un 
dicionario lingüístico, tanto para los meses y días.

### Código sin Render ###

def cantidad_filmaciones_mes(mes:str):
    if mes not in df_movie.release_date.dt.month_name(locale="es").unique():
        return f"Nombre de mes incorrecto. Datos correctos: {list(df_movie.release_date.dt.month_name(locale="es").unique())}"
    else:
        cantidad=df_movie[df_movie.release_date.dt.month_name(locale="es")==mes].shape[0]
        return f"{cantidad} películas fueron estrenadas en el mes de {mes}"

def cantidad_filmaciones_dia(dia:str):
    if dia not in df_movie.release_date.dt.day_name(locale="es").unique():
        return f"Nombre de día de semana incorrecto. Datos correctos: {list(df_movie.release_date.dt.day_name(locale="es").unique())}"
    else:
        return f"{cantidad} películas fueron estrenadas en los días {dia}"
"""

#Función 1: Cantidad de peliculas que se estrenaron por nombre de mes
@app.get("/cantidad_filmaciones_mes/{mes}",summary="Cantidad de peliculas que se estrenaron por nombre de mes")
def cantidad_filmaciones_mes(mes:str):
    # Para evitar ambigüedades, se convierten los argumentos a minúsculas en todas las funciones.
    mes=mes.lower()
    dic_month= {
    "enero": "January",
    "febrero": "February",
    "marzo": "March",
    "abril": "April",
    "mayo": "May",
    "junio": "June",
    "julio": "July",
    "agosto": "August",
    "septiembre": "September",
    "octubre": "October",
    "noviembre": "November",
    "diciembre": "December"}
    if mes not in list(dic_month.keys()):
        return {"Nombre de mes incorrecto. Datos correctos":list(dic_month.keys())}
    else:
        cantidad=df_movie[df_movie.release_date.dt.month_name()==dic_month[mes]].shape[0]
        return {"mes":mes,
                "cantidad":cantidad}

# Función 2: Cantidad de peliculas que se estrenaron por nombre del día de la semana
@app.get("/cantidad_filmaciones_dia/{dia}",summary="Cantidad de peliculas que se estrenaron por nombre del día de la semana")
def cantidad_filmaciones_dia(dia:str):
    dia=dia.lower()
    dic_week= {
    "lunes": "Monday",
    "martes": "Tuesday",
    "miércoles": "Wednesday",
    "jueves": "Thursday",
    "viernes": "Friday",
    "sábado": "Saturday",
    "domingo": "Sunday"}
    if dia not in list(dic_week.keys()):
        return {"Nombre de día de semana incorrecto. Datos correctos":list(dic_week.keys())}
    else:
        cantidad=df_movie[df_movie.release_date.dt.day_name()==dic_week[dia]].shape[0]
        return {"dia":dia,
                "cantidad":cantidad}    

# Función 3: Año de estreno y puntaje de popularidad por nombre de la filmación
"""
OBSERVACIÓN: El nombre de la película o filmación se determinará según el campo 'title'. En situaciones en las que varias películas 
compartan el mismo nombre, como en el caso de Cinderella, se seleccionará aquella que tenga el año más reciente de lanzamiento 
(release_year) tanto para la función 3 como para la 4.
"""
@app.get("/score_titulo/{titulo}",summary="Año de estreno y puntaje de popularidad por nombre de película")
def score_titulo(titulo:str):
    titulo=titulo.lower()
    if titulo not in df_movie.title.unique():
        return {"Nombre de película incorrecto. Algunos datos de ejemplo correctos":df_movie.title.iloc[:100].to_list()}
    else:
        año=int(df_movie[df_movie.title==titulo].release_year.max())
        score=float(round(df_movie[(df_movie.title==titulo)&(df_movie.release_year==año)].popularity.iloc[0],2))
        return {"titulo":titulo,
                "anio":año,
                "popularidad":score}

# Función 4:  Cantidad de votos (mayor o igual a 2000) y valor promedio de votaciones por película
@app.get("/votos_titulo/{titulo}",summary="Año de estreno, cantidad de votos (>=2000) y valor promedio de votaciones por película")
def votos_titulo(titulo:str):
    titulo=titulo.lower()
    if titulo not in df_movie.title.unique():
        return {"Nombre de película incorrecto. Algunos datos de ejemplo correctos":df_movie.title.iloc[:100].to_list()}
    else:
       año=int(df_movie[df_movie.title==titulo].release_year.max()) 
       cantidad=int(df_movie[(df_movie.title==titulo)&(df_movie.release_year==año)].vote_count.iloc[0])
       if cantidad<2000:
           return {"La película cuenta con menos de 2000 valoraciones, ingrese otro nombre. Algunas películas con más de 2000 valoraciones":
                   df_movie[df_movie.vote_count>=2000].title.iloc[:100].to_list()}
       else:
           promedio=float(round(df_movie[(df_movie.title==titulo)&(df_movie.release_year==año)].vote_average.iloc[0],2))
           return {"titulo":titulo,
                   "anio":año,
                   "voto_total":cantidad,
                   "voto_promedio":promedio}

# Función 5: Cantidad de películas, retorno total y promedio de retorno por actor (que no participó como director)
"""
OBSERVACIÓN: Tanto los actores y directores se considerarán si su nombre se encuentra en las listas correspondientes. Si el
actor también tuvo el papel de director en la misma película, como es el caso de Clint Eastwood, no se considerará la película
según las consignas.
"""
@app.get("/get_actor/{actor}",summary="Cantidad de películas, retorno total y promedio de retorno por actor")
def get_actor(actor:str):
    actor=actor.lower()
    if actor not in df_movie.actor.explode().to_list():
        return {"Nombre de actor incorrecto. Algunos datos de ejemplo correctos":df_movie.actor.explode().to_list()[:100]}
    else:
        df=df_movie[(df_movie.actor.apply(lambda x: actor in x))&(df_movie.director.apply(lambda x: actor not in x))]
        return {"actor":actor,
                "cantidad_filmaciones":df.shape[0],
                "retorno_total":float(round(df["return"].sum(),2)),
                "retorno_promedio":float(round(df['return'].mean(),2))}

# Función 6: Retorno total, nombre de cada película, fecha de lanzamiento, retorno individual, costo y ganancia de la misma por director
"""
OBSERVACIÓN: Se devolverá el retorno total por cada director, además de una lista de diccionarios con los datos pedidos de cada 
película que dirigió. Por otro lado, la ganancia se extraerá del campo 'revenue' (del inglés ganacia).
"""
@app.get("/get_director/{director}",summary="Retorno total, nombre de cada película, fecha de lanzamiento, retorno individual, costo y ganancia de la misma por director")
def get_director(director:str):
    director=director.lower()
    if director not in df_movie.director.explode().to_list():
        return {"Nombre de director incorrecto. Algunos datos de ejemplos correctos":df_movie.director.explode().to_list()[:100]}
    else:
        df=df_movie[df_movie.director.apply(lambda x: director in x)]
        lista=[]
        fecha=pd.to_datetime(df.release_date).dt.strftime("%Y-%m-%d").to_list()
        for i in range(df.shape[0]):
            dic={"película":df.title.to_list()[i],
                 "fecha de lanzamiento":fecha[i],
                 "retorno":df["return"].to_list()[i],
                 "costo":df.budget.to_list()[i],
                 "ganacia":df.revenue.to_list()[i]}
            lista.append(dic)
        return {"director":director,
                "retorno_total":float(round(df['return'].sum(),2)),
                "cantidad_peliculas":df.shape[0],
                "datos de cada película que dirigió":lista}
    
# Función 7 (ML): 5 películas con mayor puntaje (más similares) a una específica en orden descendente
@app.get("/recomendacion/{titulo}",
         summary="Cinco películas con mayor puntaje (más similares) a una específica en orden descendente")
def recomendacion(titulo:str):
    titulo=titulo.lower()
    if titulo not in df_ml.title.tolist():
        return {"Nombre de película incorrecto. Algunos datos de ejemplo correctos":list(df_ml.title)[:100]}
    else:
        indices=eval(df_ml[df_ml.title==titulo].index_movie.iloc[0])
        return {"lista recomendada":list(df_ml.title.iloc[indices].values)}