#1----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Importamos la biblioteca de Python llamada TensorFlow, que se utiliza para construir y entrenar modelos de aprendizaje automático y redes neuronales.
import tensorflow as tf

# Importamos la biblioteca de Keras desde TensorFlow. Keras es una API de alto nivel para construir y entrenar modelos de aprendizaje profundo de manera sencilla.
from tensorflow import keras

# Importamos la biblioteca NumPy para operaciones numéricas. Es útil para manejar arrays y realizar operaciones matemáticas.
import numpy as np

# Importamos la biblioteca Matplotlib para la visualización de datos. Nos permite crear gráficos y visualizar imágenes.
import matplotlib.pyplot as plt
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#2---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Imprimimos la versión de TensorFlow para confirmar que se ha instalado correctamente y verificar la versión en uso.
print(tf.__version__)
#Es importante importar las bibliotecas para poder continuar 
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#3----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Cargamos el dataset Fashion MNIST desde Keras. Este dataset contiene imágenes de artículos de moda, que se utilizarán para entrenar y evaluar el modelo.
#Aca, 60,000 imagenes son usadas para entrenar la red neuronal y 10,000 imagenes son usadas para evaluar que tan exacto aprendia la red a clasificar imagenes. 
fashion_mnist = keras.datasets.fashion_mnist
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#4----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dividimos el dataset en conjuntos de entrenamiento y prueba. 
# El conjunto de entrenamiento[train] se utiliza para entrenar el modelo y el conjunto de prueba[Test] se utiliza para evaluar su rendimiento.

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#(train_images, train_labels) es el conjunto de entrenamiento
#(test_images, test_labels) es el conjunto de prueba
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#5----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Definimos los nombres de las clases del dataset Fashion MNIST. Cada clase corresponde a un tipo de prenda de vestir.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#6-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Normalizamos las imágenes de entrenamiento para que sus valores estén en el rango de 0 a 1.
# Esto es importante para mejorar la eficiencia del entrenamiento del modelo, ya que los valores en un rango pequeño ayudan a que el modelo converja más rápido.
train_images = train_images / 255.0

# Normalizamos las imágenes de prueba de la misma manera que las imágenes de entrenamiento.
# Esto garantiza que los datos de entrada al modelo durante la evaluación tengan la misma escala que durante el entrenamiento.
test_images = test_images / 255.0
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#indice
Importación de Bibliotecas:

    TensorFlow y Keras para el aprendizaje automático.
    NumPy para cálculos matemáticos.
    Matplotlib para visualización.

Impresión de la Versión de TensorFlow:

    Verificación de la instalación.

Carga del Dataset Fashion MNIST:

    Incluye 60,000 imágenes para entrenamiento y 10,000 para prueba.

División de Datos:

    Conjunto de entrenamiento y conjunto de prueba.

Definición de Nombres de Clases:

    Nombres de las categorías de ropa.

Normalización de Imágenes:

    Escala de valores de 0 a 1 para mejorar el rendimiento del modelo.
