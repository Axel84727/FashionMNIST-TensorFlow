# Usamos el modelo para hacer predicciones sobre el conjunto de prueba.
predictions = model.predict(test_images)

# Seleccionamos una imagen específica del conjunto de prueba para mostrar cómo el modelo la clasifica.
img = test_images[5345]

# Imprimimos la forma del array de la imagen para ver sus dimensiones.
print(img.shape)

# Añadimos una dimensión adicional a la imagen para que sea compatible con la entrada del modelo.
img = (np.expand_dims(img, 0))

# Imprimimos la nueva forma del array de la imagen para confirmar que se ha añadido la dimensión extra.
print(img.shape)

# Realizamos una predicción sobre la imagen con la nueva forma.
predictions_single = model.predict(img)

# Imprimimos las predicciones para la imagen seleccionada.
print(predictions_single)

# Encontramos la etiqueta con la mayor probabilidad en las predicciones.
print(np.argmax(predictions_single[0]))
