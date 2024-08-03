import tensorflow as tf  # Importamos la biblioteca de TensorFlow y le asignamos el alias 'tf'.
from tensorflow import keras  # Importamos el módulo 'keras' desde TensorFlow.

def load_fashion_mnist_data():  # Definimos la función 'load_fashion_mnist_data' para cargar y procesar los datos del dataset.
    fashion_mnist = keras.datasets.fashion_mnist  # Accedemos al dataset Fashion MNIST desde Keras.
    
    # Cargamos el dataset Fashion MNIST.
    # El dataset se divide en dos partes:
    # (train_images, train_labels): Imágenes y etiquetas para el conjunto de entrenamiento.
    # (test_images, test_labels): Imágenes y etiquetas para el conjunto de prueba.
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  
    
    # Definimos los nombres de las clases del dataset Fashion MNIST.
    # Estos nombres representan las categorías de las imágenes en el dataset.
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Normalizamos las imágenes de entrenamiento.
    # Dividimos los valores de los píxeles por 255 para que estén en el rango de 0 a 1.
    # Esto ayuda a que el modelo entrene más eficientemente.
    train_images = train_images / 255.0
    
    # Normalizamos las imágenes de prueba de la misma manera que las imágenes de entrenamiento.
    # Esto asegura que los datos de prueba también estén en el rango de 0 a 1.
    test_images = test_images / 255.0
