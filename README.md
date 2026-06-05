# Proyecto - Steam Game MetaData - Tags vs Éxito

## Descripción

Este repositorio contiene el desarrollo del proyecto de análisis de datos enfocado en videojuegos de Steam, cuyo objetivo es estudiar la relación entre los **tags de los juegos** y su nivel de éxito dentro de la plataforma.

El proyecto utiliza técnicas de:

- Análisis Exploratorio de Datos (EDA)
- Visualización de datos
- Limpieza y procesamiento de datos
- Análisis de correlación y patrones

para responder la problemática principal:

> **¿Será un juego exitoso en base a sus tags?**

Dataset utilizado:

**Steam Game Dataset**

Dataset obtenido desde Kaggle:  
https://www.kaggle.com/datasets/newnguyn/steam-game-clean

---

## Objetivo del proyecto

Analizar metadata de videojuegos de Steam para identificar patrones relacionados con el éxito de un juego, utilizando variables como:

- Tags
- Reviews positivas
- Review Score
- Precio
- Fecha de lanzamiento
- Popularidad

A través del análisis se busca:

- Detectar tags asociados al éxito.
- Identificar tendencias del mercado de videojuegos.
- Analizar distribuciones y comportamiento de reviews.
- Detectar outliers y comportamientos atípicos.
- Obtener insights útiles para futuros modelos predictivos.

---

## Tecnologías utilizadas

- Python 3
- Jupyter Notebook / Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly

---

## Estructura del proyecto

```bash
Steam_Game_MetaData/
│
├── data/                  # Dataset original y archivos procesados
├── notebook/              # Notebooks del análisis
│   └── steam_analysis.ipynb
│
├── images/                # Gráficos y visualizaciones
├── src/                   # Scripts auxiliares
├── requirements.txt
└── README.md
```

## Instalación

Clonar repositorio:

```bash
git clone <url-del-repositorio>
cd Tarea6-Clasificacion
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Ejecución

Ejecutar notebook:

```bash
jupyter notebook
```

o abrir directamente en Google Colab.

---

## Integrantes

- Ignacio Manríquez ([@lNachox](https://github.com/lNachox))
- Carlos Cienfuegos ([@CarlosCienfuegos1](https://github.com/CarlosCienfuegos1))
- Eduardo Krause ([@Eduardok01](https://github.com/Eduardok01))
- Cristopher Gallegos ([@Ertrax147](https://github.com/Ertrax147))

---

## Asignatura

**ICC732-1: Ingeniería de Datos**  
**Universidad de la Frontera
