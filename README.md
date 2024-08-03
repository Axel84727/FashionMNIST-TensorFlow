# Fashion MNIST Classifier

## Descripción

Este proyecto utiliza el dataset Fashion MNIST para construir, entrenar y evaluar un modelo de clasificación de imágenes. El objetivo es clasificar imágenes de artículos de moda en una de diez categorías diferentes utilizando un modelo de red neuronal simple.

## Estructura del Proyecto

La estructura del proyecto es la siguiente:

- **`data/`**: Contiene scripts para cargar y preprocesar los datos.
  - `load_data.py`: Código para cargar el dataset Fashion MNIST y normalizar las imágenes.
- **`model/`**: Contiene scripts para construir, entrenar y evaluar el modelo.
  - `build_model.py`: Definición de la arquitectura del modelo.
  - `train_model.py`: Código para entrenar el modelo con los datos de entrenamiento.
  - `evaluate_model.py`: Código para evaluar el rendimiento del modelo con los datos de prueba.
- **`scripts/`**: Contiene scripts para la visualización de los resultados.
  - `plot_utils.py`: Funciones para visualizar imágenes y gráficas de predicciones.
- **`utils/`**: (Opcional) Utilidades generales para el proyecto.
- **`main.py`**: Archivo principal para ejecutar el flujo del proyecto.
- **`README.md`**: Este archivo, que proporciona una visión general del proyecto.

## Requisitos

Para ejecutar este proyecto, necesitarás instalar las siguientes bibliotecas:

- TensorFlow
- NumPy
- Matplotlib

Puedes instalar las dependencias utilizando `pip`:

```bash
pip install tensorflow numpy matplotlib
