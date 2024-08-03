
# Importamos TensorFlow y Keras para construir y definir el modelo de red neuronal.
import tensorflow as tf
from tensorflow import keras

def create_model():
    """
    Define y compila un modelo de red neuronal para clasificar imágenes de moda.
    
    Returns:
        model: Un modelo de red neuronal compilado listo para ser entrenado.
    """
    # Creamos un modelo secuencial, que es una pila lineal de capas.
    model = keras.Sequential([
        # La primera capa aplana la imagen de 28x28 píxeles a un vector de 784 elementos.
        # Esto convierte la imagen en una entrada unidimensional (una unica dimesion) para las siguientes capas densas.
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
