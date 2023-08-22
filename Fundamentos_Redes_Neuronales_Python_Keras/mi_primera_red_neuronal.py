import numpy as np
from keras import layers, models
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt


# Se carga dataset
(train_data, train_labels) , (test_data, test_labels) = mnist.load_data()

# Definición del modelo
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(10, activation='softmax'))

# Compilacion de la red
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics='accuracy')

# Muestra la arquitectura del modelo actual
model.summary()
print("")

# Modificamos el dataset de entrenamiento. Pasamos de 3d a 2d
x_train = train_data.reshape((60000,28*28))
x_train = x_train.astype('float32')/255

x_test = test_data.reshape((10000,28*28))
x_test = x_test.astype('float32')/255

# Hacemos One Hot Encoding
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)


# Entrenar la red
model.fit(x_train, y_train, epochs=15, batch_size=128)

# Evaluamos
model.evaluate(x_test, y_test)