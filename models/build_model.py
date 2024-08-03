def build_model():  # Definimos la función 'build_model' para construir y compilar un modelo de red neuronal.
    # Creamos un modelo secuencial usando Keras. 
    # Un modelo secuencial es una pila lineal de capas, donde cada capa tiene una única entrada y salida.
    model = keras.Sequential([
        # La primera capa 'Flatten' convierte las imágenes de entrada de 2D (28x28 píxeles) en un vector 1D de 784 píxeles.
        # Esto es necesario porque las siguientes capas densas requieren entradas unidimensionales.
        keras.layers.Flatten(input_shape=(28, 28)),
        
        # La segunda capa es una capa densa con 128 neuronas.
        # La función de activación 'relu' (Rectified Linear Unit) introduce no linealidad en el modelo,
        # permitiendo que el modelo aprenda patrones más complejos.
        keras.layers.Dense(128, activation='relu'),
        
        # La última capa es una capa densa con 10 neuronas.
        # La función de activación 'softmax' convierte las salidas en probabilidades que suman 1.
        # Cada neurona representa una clase, y la capa final produce una probabilidad para cada clase.
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compilamos el modelo.
    # Especificamos el optimizador, la función de pérdida y las métricas a utilizar durante el entrenamiento.
    model.compile(optimizer='adam',  # 'adam' es un optimizador que ajusta los pesos del modelo durante el entrenamiento.
                  loss='sparse_categorical_crossentropy',  # La función de pérdida 'sparse_categorical_crossentropy' se usa para clasificación múltiple con etiquetas enteras.
                  metrics=['accuracy'])  # 'accuracy' es la métrica que se usará para evaluar el rendimiento del modelo.
    
    return model  # Devolvemos el modelo compilado.
