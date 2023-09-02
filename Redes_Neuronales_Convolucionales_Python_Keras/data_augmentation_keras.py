from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt

print("Hola")

# Data Generator
datagen = ImageDataGenerator(rotation_range=40,
							width_shift_range=0.2,
							height_shift_range=0.2,
							zoom_range=0.2,
							horizontal_flip=True,
							fill_mode='nearest',
							brightness_range=[0.4,1.5])

# Generar basado en array/imagen
img = load_img('./data/jirafa.jpg')
x = img_to_array(img)
print(x.shape)
x = x.reshape((1,)+ x.shape)
print(x.shape)

# Data generator basado en un directorio
# i = 0 
# for batch in datagen.flow(x, batch_size=1):
#     plt.figure(i)
#     imgplot = plt.imshow(array_to_img(batch[0]))
#     i += 1
#     if i % 10 == 0 :
#         break
# plt.show()

train_generator = datagen.flow_from_directory('./data/cats_and_dogs/train',
												target_size=(150,150),
												batch_size=32,
												class_mode='binary'
 											)

print(train_generator[0][0].shape)

img = array_to_img(train_generator[0][0][1])

plt.imshow(img)

plt.show()

