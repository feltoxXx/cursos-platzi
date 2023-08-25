import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import models, layers, optimizers
from keras import regularizers

# Descargamos los datos de imdb - Keras
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# Función de one-hot encoding
def vectorizar(sequences, dim=10000):
	results = np.zeros((len(sequences), dim))
	for i, sequences in enumerate(sequences):
		results[i, sequences]= 1
	return results

# Diccionario de palabras
word_index = imdb.get_word_index()
word_index = dict([(value,key) for (key,value) in word_index.items()])


# for _ in train_data[0]:
# 	print(word_index.get(_ - 3))

# Transformar la data para ingresar a keras
x_train = vectorizar(train_data)
x_test = vectorizar(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# Preparamos la data
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train =  y_train[10000:]

# Creamos el modelo 1
model1 = models.Sequential()
model1.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model1.add(layers.Dense(16, activation='relu'))
model1.add(layers.Dense(1, activation='sigmoid'))

# Compilamos el modelo 1
model1.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
             metrics=['acc'])

# Entrenamos el modelo 1
history1 = model1.fit(partial_x_train,
                   partial_y_train,
                   epochs=20,
                   batch_size=512,
                   validation_data=(x_val,y_val))


## Modelo menos complejo
# Creamos el modelo 2
model2 = models.Sequential()
model2.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
model2.add(layers.Dense(4, activation='relu'))
model2.add(layers.Dense(1, activation='sigmoid'))

# Compilamos el modelo 2
model2.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
             metrics=['acc'])

# Entrenamos el modelo 2
history2 = model2.fit(partial_x_train,
                   partial_y_train,
                   epochs=20,
                   batch_size=512,
                   validation_data=(x_val,y_val))


## Regularización
# Creamos el modelo 3
model3 = models.Sequential()
model3.add(layers.Dense(16, activation='relu', input_shape=(10000,), kernel_regularizer=regularizers.l2(0.001)))
model3.add(layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model3.add(layers.Dense(1, activation='sigmoid'))

# Compilamos el modelo 3
model3.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
             metrics=['acc'])

# Entrenamos el modelo 3
history3 = model3.fit(partial_x_train,
                   partial_y_train,
                   epochs=20,
                   batch_size=512,
                   validation_data=(x_val,y_val))

## Dropout
# Creamos el modelo 4
model4 = models.Sequential()
model4.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model4.add(layers.Dropout(0.5))
model4.add(layers.Dense(16, activation='relu'))
model4.add(layers.Dropout(0.5))
model4.add(layers.Dense(1, activation='sigmoid'))

# Compilamos el modelo 4
model4.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
             metrics=['acc'])

# Entrenamos el modelo 4
history4 = model4.fit(partial_x_train,
                   partial_y_train,
                   epochs=20,
                   batch_size=512,
                   validation_data=(x_val,y_val))


# Analizamos resultados
history_dict = history1.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

val_loss_values2 = history2.history['val_loss']
val_loss_values3 = history3.history['val_loss']
val_loss_values4 = history4.history['val_loss']


fig = plt.figure(figsize=(10,10))
epoch = range(1,len(loss_values)+1)
plt.plot(epoch,val_loss_values2, 'o',label='smaller')
plt.plot(epoch,val_loss_values, '--',label='original')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,10))
epoch = range(1,len(loss_values)+1)
plt.plot(epoch,val_loss_values3, 'o',label='regularization')
plt.plot(epoch,val_loss_values, '--',label='original')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,10))
epoch = range(1,len(loss_values)+1)
plt.plot(epoch,val_loss_values4, 'o',label='dropout')
plt.plot(epoch,val_loss_values, '--',label='original')
plt.legend()
plt.show()


model1.evaluate(x_test, y_test)
model2.evaluate(x_test, y_test)
model3.evaluate(x_test, y_test)
model4.evaluate(x_test, y_test)


# # Predicciones
# predictions = model.predict(x_test)
# predictions[1]