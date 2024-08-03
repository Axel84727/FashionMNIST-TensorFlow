#1----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Importamos la biblioteca de Python llamada TensorFlow, que se utiliza para construir y entrenar modelos de aprendizaje automático y redes neuronales.
import tensorflow as tf

# Importamos la biblioteca de Keras desde TensorFlow. Keras es una API de alto nivel para construir y entrenar modelos de aprendizaje profundo de manera sencilla.
from tensorflow import keras

# Importamos la biblioteca NumPy para operaciones numéricas. Es útil para manejar arrays y realizar operaciones matemáticas.
import numpy as np

# Importamos la biblioteca Matplotlib para la visualización de datos. Nos permite crear gráficos y visualizar imágenes.
import matplotlib.pyplot as plt

# Importamos la función load_data desde el archivo load_data.py
from load_data import load_data
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#7----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def create_model():
    """
    Define y compila un modelo de red neuronal para clasificar imágenes de moda.
    Retorna el modelo compilado.
    """
    # Creamos un modelo secuencial, que es una pila lineal de capas.
    model = keras.Sequential([
        # La primera capa aplana la imagen de 28x28 píxeles a un vector de 784 elementos.
        # Esto convierte la imagen en una entrada unidimensional para las siguientes capas densas.
        keras.layers.Flatten(input_shape=(28, 28)),
        
        # Añadimos una capa densa con 128 neuronas y función de activación ReLU.
        # La activación ReLU introduce no linealidad en el modelo permitiendo al modelo aprender patrones complejos.
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
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#8----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def train_model(train_images, train_labels, epochs=5):
    """
    Entrena el modelo con el conjunto de datos de entrenamiento.
    """
    # Creamos el modelo.
    model = create_model()
    
    # Entrenamos el modelo usando los datos de entrenamiento.
    # 'epochs' indica el número de veces que el modelo pasará por todo el conjunto de datos.
    model.fit(train_images, train_labels, epochs=epochs)
    
    return model
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#9----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Código para ejecutar el entrenamiento si se ejecuta este archivo directamente.
if __name__ == "__main__":
    import argparse

    # Definimos el argumento para el número de épocas desde la línea de comandos.
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Número de épocas para entrenar el modelo')
    args = parser.parse_args()

    # Cargamos los datos.
    train_images, train_labels, test_images, test_labels = load_data()
    
    # Entrenamos el modelo.
    model = train_model(train_images, train_labels, epochs=args.epochs)
    
    # Guardamos el modelo entrenado.
    model.save('fashion_mnist_model.h5')
    print("Modelo guardado como 'fashion_mnist_model.h5'")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
