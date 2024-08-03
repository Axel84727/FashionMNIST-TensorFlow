def train_model():
    (train_images, train_labels), (test_images, test_labels) = load_fashion_mnist_data()
    model = build_model()
    model.fit(train_images, train_labels, epochs=5)
