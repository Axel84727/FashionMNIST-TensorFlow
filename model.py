# Importamos la biblioteca de Python llamada TensorFlow, que se utiliza para construir y entrenar modelos de aprendizaje automático y redes neuronales.
import tensorflow as tf
from tensorflow import keras

# Importamos la biblioteca NumPy para operaciones numéricas.
import numpy as np

# Importamos la biblioteca Matplotlib para la visualización de datos.
import matplotlib.pyplot as plt

"""
Define y compila un modelo de red neuronal para clasificar imágenes de moda.
"""
def create_model():
    # Creamos un modelo secuencial, que es una pila lineal de capas.
    model = keras.Sequential([
        # La primera capa aplana la imagen de 28x28 píxeles a un vector de 784 elementos.
        # Esto convierte la imagen en una entrada unidimensional para las siguientes capas densas.
        keras.layers.Flatten(input_shape=(28, 28)),
        
        # Añadimos una capa densa con 128 neuronas y función de activación ReLU.
        # La activación ReLU introduce no linealidad en el modelo permitiendo al modelo salir de las operaciones lineales.
        keras.layers.Dense(128, activation='relu'),
        
        # Añadimos la capa de salida con 10 neuronas (una por cada clase en Fashion MNIST).
        # Usamos la función de activación softmax para obtener probabilidades de clase.
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compilamos el modelo especificando el optimizador, la función de pérdida y las métricas.
    # El optimizador 'adam' ajusta los pesos del modelo durante el entrenamiento.
    # La función de pérdida 'sparse_categorical_crossentropy' calcula la diferencia entre las predicciones y las etiquetas verdaderas.
    # La métrica 'accuracy' se usa para evaluar la precisión del modelo.
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model(train_images, train_labels):
    # Crear el modelo.
    model = create_model()
    
    # Entrenamos el modelo con el conjunto de entrenamiento.
    # Las epochs son la cantidad de vueltas que da el modelo sobre los datos de entrenamiento.
    model.fit(train_images, train_labels, epochs=5)
    
    # Guardamos el modelo entrenado para su uso futuro.
    model.save('fashion_mnist_model.h5')
    
    return model

# Código para ejecutar el entrenamiento si se ejecuta este archivo directamente.
if __name__ == "__main__":
    # Importamos las funciones para cargar datos.
    from load_data import load_data
    
    # Cargamos los datos.
    train_images, train_labels, test_images, test_labels = load_data()
    
    # Entrenamos el modelo.
    model = train_model(train_images, train_labels)
