# Importando librerías
import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.models import clone_model

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Cargando dataset de Fashion MNIST
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape)
# plt.imshow(train_images[99])
# plt.show()

# Limpieza de datos
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)


# Creando nuestra red neuronal¶
model = tf.keras.Sequential()
# Must define the input shape in the first layer of the neural network
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# Take a look at the model summary
model.summary()

# Compilando la red neuronal
model.compile(loss='categorical_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])

# Entenando la red
model.fit(train_images,
         train_labels,
         batch_size=64,
         epochs=10)

# Análisis de resultados
score = model.evaluate(test_images, test_labels, verbose=0)
print(f'score: {score}')

# Callbacks
early = tf.keras.callbacks.EarlyStopping(monitor='accuracy',patience=1)
model.fit(train_images,
          train_labels,
          batch_size=64,
          callbacks=[early],
          epochs=10)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='./models/mi_primer_red_conv.hdf5',
                                verbose=1,
                                monitor='accuracy',
                                save_best_only=True)
model.fit(train_images,
         train_labels,
         batch_size=64,
         callbacks=[checkpoint],
         epochs=10)

model2 = clone_model(model)
model2.load_weights('./mi_primer_red_conv.hdf5')

model2.evaluate(test_images, test_labels)