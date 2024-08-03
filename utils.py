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
