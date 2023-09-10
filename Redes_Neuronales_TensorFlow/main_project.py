
import matplotlib.pyplot as plt
import string
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.datasets import cifar100


(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Limpieza de datos
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


train_dir = "./data/sign-language-img/Train"
test_dir = "./data/sign-language-img/Test"

train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255, validation_split= 0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size= (28,28),
    batch_size= 128,
    class_mode="categorical",
    color_mode="grayscale",
    subset="training"
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size= (28,28),
    batch_size= 128,
    class_mode="categorical",
    color_mode="grayscale",
    subset="validation"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size= (28,28),
    batch_size= 128,
    class_mode="categorical",
    color_mode="grayscale"
)

classes = [char for char in string.ascii_uppercase if char != "J" if char != "Z"]
print(classes)
print("")

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize = (10, 10))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img[:,:,0])
        ax.axis("off")
    plt.tight_layout()
    plt.show()

sample_training_images, _ = next(train_generator)
plotImages(sample_training_images[:5])

model_base = Sequential()

model_base.add(Flatten(input_shape = (28, 28, 1)))
model_base.add(Dense(256, activation= "relu"))
model_base.add(Dense(256, activation= "relu"))
model_base.add(Dense(128, activation= "relu"))
model_base.add(Dense(len(classes), activation= "softmax"))

model_base.summary()

model_base.compile(loss= "categorical_crossentropy",
                   optimizer= optimizers.Adam(),
                   metrics= ['accuracy'])


history =model_base.fit(
    train_generator,
    epochs=20,
    validation_data= validation_generator
)