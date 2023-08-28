import numpy as np 
import matplotlib.pyplot as plt
from skimage import io

# Cargando imágenes
im = io.imread('./data/jirafa.jpg')
print(im.shape)

# plt.imshow(im)
# plt.show()

# Separando canales de color
r = im[:,:,0]
g = im[:,:,1]
b = im[:,:,2]

# plt.imshow(r.T,cmap='gray')
# plt.show()

# RGB con un único canal activo
aux_dim = np.zeros([1080,1920])

red = np.dstack((r,aux_dim, aux_dim)).astype(np.uint8)
green = np.dstack((aux_dim, g, aux_dim)).astype(np.uint8)
blue = np.dstack((aux_dim, aux_dim,b)).astype(np.uint8)

# plt.imshow(red)
# plt.show()
# plt.imshow(green)
# plt.show()
# plt.imshow(blue)
# plt.show()

all_channels = np.concatenate((red, green, blue), axis=1)
# plt.imshow(all_channels)
# plt.show()

# Otros cambios y modificaciones a imágenes
im_neg_pos = 255 - im
im32 = (im//32) * 32
im128 = (im//128) * 128

# plt.imshow(im128)
# plt.show()

plt.imshow(im[0:200,1040:1300])
plt.show()

plt.imshow(np.concatenate((im , im_neg_pos, im32, im128), axis=1))
plt.show()