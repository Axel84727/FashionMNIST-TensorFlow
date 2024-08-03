def train_model():  # Definimos la función 'train_model' para entrenar el modelo de red neuronal.
    # Llamamos a la función 'load_fashion_mnist_data' para cargar los datos del dataset.
    # Esta función devuelve los conjuntos de imágenes y etiquetas para entrenamiento y prueba.
    (train_images, train_labels), (test_images, test_labels) = load_fashion_mnist_data()
    
    # Llamamos a la función 'build_model' para crear un modelo de red neuronal.
    # La función 'build_model' define la estructura del modelo y lo compila.
    model = build_model()
    
    # Entrenamos el modelo con los datos de entrenamiento.
    # 'model.fit' ajusta el modelo a los datos de entrenamiento.
    # El parámetro 'epochs=5' especifica que el entrenamiento se realizará durante 5 épocas.
    model.fit(train_images, train_labels, epochs=5)
    
    # La función no devuelve nada, pero entrena el modelo en los datos proporcionados.
