# Importamos las bibliotecas necesarias
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Función para cargar y normalizar los datos
def load_data():
    """
    Carga y normaliza el dataset Fashion MNIST.
    Retorna los datos de entrenamiento y prueba.
    """
    # Imprimimos la versión de TensorFlow
    print(tf.__version__)

    # Cargamos el dataset Fashion MNIST desde Keras
    fashion_mnist = keras.datasets.fashion_mnist

    # Dividimos el dataset en conjuntos de entrenamiento y prueba
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalizamos las imágenes de entrenamiento y prueba
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return train_images, train_labels, test_images, test_labels

# Definimos los nombres de las clases del dataset Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Función para crear y compilar el modelo
def create_model():
    """
    Define y compila un modelo de red neuronal para clasificar imágenes de moda.
    Retorna el modelo compilado.
    """
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Función para entrenar el modelo
def train_model(train_images, train_labels):
    """
    Entrena el modelo con el conjunto de datos de entrenamiento.
    """
    model = create_model()
    model.fit(train_images, train_labels, epochs=5)
    return model

# Función para evaluar el modelo
def evaluate_model():
    """
    Evalúa el modelo de red neuronal usando el dataset de prueba.
    """
    # Cargamos los datos de prueba
    _, _, test_images, test_labels = load_data()
    
    # Cargamos el modelo entrenado
    model = keras.models.load_model('fashion_mnist_model.h5')
    
    # Realizamos predicciones sobre todas las imágenes de prueba
    predictions = model.predict(test_images)
    
    # Seleccionamos una imagen de prueba para analizar
    img = test_images[0]
    print(img.shape)
    
    # Expandimos las dimensiones de la imagen
    img = np.expand_dims(img, 0)
    print(img.shape)
    
    # Realizamos una predicción sobre la imagen seleccionada
    predictions_single = model.predict(img)
    print(predictions_single)
    print(np.argmax(predictions_single[0]))

# Función para visualizar una imagen con su predicción y etiqueta verdadera
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                          100 * np.max(predictions_array),
                                          class_names[true_label]),
               color=color)

# Función para visualizar un gráfico de barras con las probabilidades de las predicciones
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

if __name__ == "__main__":
    # Cargamos los datos
    train_images, train_labels, test_images, test_labels = load_data()
    
    # Entrenamos el modelo
    model = train_model(train_images, train_labels)
    
    # Guardamos el modelo entrenado
    model.save('fashion_mnist_model.h5')
    print("Modelo guardado como 'fashion_mnist_model.h5'")
    
    # Evaluamos el modelo
    evaluate_model()

    # Visualización de predicciones
    predictions = model.predict(test_images)
    i = 0  # Modifica este número para ver la predicción y el porcentaje de acierto de otra imagen
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()
