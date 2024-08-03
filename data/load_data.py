import tensorflow as tf  # Importa la biblioteca TensorFlow y la asigna al alias 'tf'.
from tensorflow import keras  # Importa el módulo 'keras' desde TensorFlow para construir y entrenar modelos de redes neuronales.


    fashion_mnist = keras.datasets.fashion_mnist  # Accede al dataset Fashion MNIST desde Keras.
    
    # Carga los datos del dataset:
    # - train_images y train_labels contienen las imágenes y etiquetas para el conjunto de entrenamiento.
    # - test_images y test_labels contienen las imágenes y etiquetas para el conjunto de prueba.
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  
    
    # Define los nombres de las clases del dataset Fashion MNIST.
    # Cada nombre representa una categoría de prenda de vestir en el dataset.
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Normaliza las imágenes del conjunto de entrenamiento.
    # Los valores de píxeles van de 0 a 255, y dividir por 255.0 convierte estos valores al rango de 0 a 1.
    # Esto ayuda a que el modelo entrene de manera más eficiente.
    train_images = train_images / 255.0
    
    # Normaliza las imágenes del conjunto de prueba de la misma manera que las imágenes de entrenamiento.
    # Esto asegura que los datos de entrada al modelo durante la evaluación tengan la misma escala.
    test_images = test_images / 255.0
