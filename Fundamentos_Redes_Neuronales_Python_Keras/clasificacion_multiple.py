import numpy as np
from keras import layers, models
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt 


# Descarguemos datos
(train_data, train_labels) , (test_data, test_labels) = reuters.load_data(num_words=10000)

# Diccionario de palabras
word_index = reuters.get_word_index()
word_index = dict([(value,key) for (key,value) in word_index.items()])

# for _ in train_data[0]:
# 	print(word_index.get(_ - 3))

# Función de vectorizar
def vectorizar(sequences, dim=10000):
    results = np.zeros((len(sequences),dim))
    for i, sequences in enumerate(sequences):
        results[i,sequences]=1
    return results

# Transformando los datos
x_train = vectorizar(train_data)
x_test = vectorizar(test_data)

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

# Creando la red
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(46, activation='softmax'))

# Compilando la red
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc']   
             )

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train[:1000]
partial_y_train =  y_train[1000:]


# Entrenando el modelo
history = model.fit(partial_x_train,
                   partial_y_train,
                   epochs=20,
                   batch_size=128,
                   validation_data=(x_val,y_val))

# Validamos resultados
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

predictions = model.predict(x_test)

predictions[0]

print(np.sum(predictions[0]))

print(np.argmax([2,5,65,3,6,7]))

print(np.argmax(predictions[0]))