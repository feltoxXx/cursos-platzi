import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import models, layers, optimizers

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


# Creamos el modelo
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compilamos el modelo
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
             metrics=['acc'])

# Entrenamos el modelo
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train =  y_train[10000:]

history = model.fit(partial_x_train,
                   partial_y_train,
                   epochs=20,
                   batch_size=512,
                   validation_data=(x_val,y_val))

# Analizamos resultados
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

fig = plt.figure(figsize=(10,10))
epoch = range(1,len(loss_values)+1)
plt.plot(epoch,loss_values, 'o',label='training')
plt.plot(epoch,val_loss_values, '--',label='val')
plt.legend()
plt.show()

model.evaluate(x_test, y_test)


# # Predicciones
# predictions = model.predict(x_test)
# predictions[1]