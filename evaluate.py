# Importamos TensorFlow y Keras para trabajar con modelos de aprendizaje automático.
import tensorflow as tf
from tensorflow import keras

# Importamos NumPy para operaciones numéricas.
import numpy as np

# Importamos la función para cargar datos desde el archivo load_data.py.
from load_data import load_data

def evaluate_model():
    """
    Evalúa el modelo de red neuronal usando el dataset de prueba.
    """
    # Cargamos los datos de prueba desde load_data.py.
    _, _, test_images, test_labels = load_data()
    
    # Cargamos el modelo entrenado guardado previamente.
    model = keras.models.load_model('fashion_mnist_model.h5')
    
    # Realizamos predicciones sobre todas las imágenes de prueba.
    predictions = model.predict(test_images)
    
    # Seleccionamos una imagen de prueba para analizar.
    img = test_images[0]
    
    # Imprimimos la forma de la imagen original (28x28 píxeles).
    print(img.shape)
    
    # Expandimos las dimensiones de la imagen para que sea compatible con el modelo.
    img = np.expand_dims(img, 0)
    
    # Imprimimos la forma de la imagen después de expandir las dimensiones (1x28x28).
    print(img.shape)
    
    # Realizamos una predicción sobre la imagen seleccionada.
    predictions_single = model.predict(img)
    
    # Imprimimos las predicciones para la imagen seleccionada.
    print(predictions_single)
    
    # Imprimimos la etiqueta de clase con la mayor probabilidad para la imagen seleccionada.
    print(np.argmax(predictions_single[0]))

if __name__ == "__main__":
    # Ejecutamos la función evaluate_model si este archivo es ejecutado directamente.
    evaluate_model()
