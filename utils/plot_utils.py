# Definimos una función para mostrar la imagen con la predicción del modelo y la etiqueta real.
def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)  # Desactiva la cuadrícula del gráfico.
    plt.xticks([])  # Elimina las marcas del eje x.
    plt.yticks([])  # Elimina las marcas del eje y.

    plt.imshow(img, cmap=plt.cm.binary)  # Muestra la imagen en escala de grises.

    predicted_label = np.argmax(predictions_array)  # Encuentra la etiqueta con la mayor probabilidad.
    color = 'blue' if predicted_label == true_label else 'red'  # Elige el color en función de si la predicción es correcta o no.

    # Muestra la etiqueta predicha y la etiqueta verdadera en la imagen.
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                          100*np.max(predictions_array),
                                          class_names[true_label]),
                                          color=color)

# Definimos una función para mostrar un gráfico de las probabilidades de cada clase.
def plot_value_array(i, predictions_array, true_label, class_names):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)  # Desactiva la cuadrícula del gráfico.
    plt.xticks(range(10))  # Muestra las marcas del eje x.
    plt.yticks([])  # Elimina las marcas del eje y.
    thisplot = plt.bar(range(10), predictions_array, color="#777777")  # Crea un gráfico de barras con las probabilidades.
    plt.ylim([0, 1])  # Establece el límite del eje y.

    predicted_label = np.argmax(predictions_array)  # Encuentra la etiqueta con la mayor probabilidad.

    # Cambia el color de las barras para la etiqueta predicha y la etiqueta verdadera.
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Usamos la función plot_value_array para mostrar un gráfico de las probabilidades de la primera predicción.
plot_value_array(1, predictions_single[0], test_labels, class_names)
_ = plt.xticks(range(10), class_names, rotation=45)  # Añade nombres de clases al gráfico y los rota.

