from tensorflow import keras
from keras.utils import to_categorical
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape)
# print(x_test.shape)

# plt.imshow(x_train[6])
# plt.show()

# Limpieza de datos
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

num_clases = len(np.unique(y_train))
y_train = to_categorical(y_train, num_clases)
y_test = to_categorical(y_test, num_clases)

# print(y_train[0])

# Dividimos el dataset para tener validación
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

print(f'x_train: {x_train.shape}')
print(f'train: {x_train.shape[0]}')
print(f'valid: {x_valid.shape[0]}')
print(f'test: {x_test.shape[0]}')

# Creamos el modelo
base_filtros = 32
w_regularizer = 1e-4
input_shape = x_train.shape[1:]

model = Sequential()
# Convolución 1
model.add(Conv2D(base_filtros, (3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizer), input_shape = input_shape))
model.add(Activation('relu'))

# Convolución 2
model.add(Conv2D(base_filtros, (3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizer)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Convolución 3
model.add(Conv2D(2*base_filtros, (3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizer)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Convolución 4
model.add(Conv2D(2*base_filtros, (3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizer)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

# Convolución 5
model.add(Conv2D(4*base_filtros, (3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizer)))
model.add(Activation('relu'))

# Convolución 6
model.add(Conv2D(4*base_filtros, (3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizer)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

# Clasificación - Flatten
model.add(Flatten())
model.add(Dense(num_clases,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_valid, y_valid), verbose=2, shuffle=True)

plt.plot(hist.history['accuracy'],label='Train')
plt.plot(hist.history['val_accuracy'],label='Valid')
plt.legend()
plt.show()

checkpoint = keras.callbacks.ModelCheckpoint(filepath='cifar_clasificacion.hdf5',
                                            verbose=1,
                                            monitor='accuracy',
                                            save_best_only=True)

model.evaluate(x_test,y_test)