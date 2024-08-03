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

def train_model(train_images, train_labels):
    """
    Entrena el modelo con el conjunto de datos de entrenamiento.
    """
    # Creamos el modelo.
    model = create_model()
    
    # Entrenamos el modelo usando los datos de entrenamiento.
    # 'epochs' indica el número de veces que el modelo pasará por todo el conjunto de datos.
    model.fit(train_images, train_labels, epochs=5)
    
    return model

# Código para ejecutar el entrenamiento si se ejecuta este archivo directamente.
if __name__ == "__main__":
    # Cargamos los datos.
    train_images, train_labels, test_images, test_labels = load_data()
    
    # Entrenamos el modelo.
    model = train_model(train_images, train_labels)
    
    # Guardamos el modelo entrenado.
    model.save('fashion_mnist_model.h5')
    print("Modelo guardado como 'fashion_mnist_model.h5'")

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

# Importamos NumPy para manejar arrays y realizar operaciones matemáticas.
import numpy as np

# Importamos Matplotlib para la visualización de datos.
import matplotlib.pyplot as plt

# Definimos una función para visualizar una imagen con su predicción y etiqueta verdadera.
def plot_image(i, predictions_array, true_label, img):
    # Extraemos la predicción y la etiqueta verdadera para la imagen en la posición 'i'.
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    
    # Configuramos el gráfico para no mostrar la cuadrícula y ocultar las marcas de los ejes.
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    # Mostramos la imagen en escala de grises.
    plt.imshow(img, cmap=plt.cm.binary)
    
    # Determinamos la etiqueta predicha con la mayor probabilidad.
    predicted_label = np.argmax(predictions_array)
    
    # Definimos el color de la etiqueta en función de si la predicción es correcta o no.
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    # Mostramos el nombre de la etiqueta predicha y la etiqueta verdadera, con el porcentaje de certeza.
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                          100 * np.max(predictions_array),
                                          class_names[true_label]),
               color=color)

# Definimos una función para visualizar un gráfico de barras con las probabilidades de las predicciones.
def plot_value_array(i, predictions_array, true_label):
    # Extraemos la predicción y la etiqueta verdadera para la imagen en la posición 'i'.
    predictions_array, true_label = predictions_array, true_label[i]
    
    # Configuramos el gráfico para no mostrar la cuadrícula y mostrar marcas en el eje x.
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    
    # Creamos un gráfico de barras con las probabilidades de cada clase.
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    
    # Determinamos la etiqueta con la mayor probabilidad.
    predicted_label = np.argmax(predictions_array)
    
    # Cambiamos el color de la barra para la etiqueta predicha y la etiqueta verdadera.
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Ejemplo de uso para visualizar las predicciones.
# (Este bloque de código puede ser movido a una función principal si se deseas ejecutarlo directamente.)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

i = 0 #modificas este numero por la imagen que deseas ver su prediccion y porcentaje de acierto 
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()
