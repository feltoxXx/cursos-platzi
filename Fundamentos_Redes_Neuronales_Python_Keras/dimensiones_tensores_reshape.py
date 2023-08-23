import numpy as np


# Escalar
x = np.array(42)
print(x.shape)
print(x.ndim)

# Vector
x = np.array([42,23,453,21,32,1,3,7])
print("")
print(x.shape)
print(x.ndim)

# Matriz
print("")
x = np.array([[42,23,453],
             [34,343,24]])
print(x.shape)
print(x.ndim)

# Tensor
print("")
x = np.array([[[42,23,453],
             [34,343,24]],
             [[42,23,453],
             [34,343,24]],
             [[42,23,453],
             [34,343,24]],
             [[42,23,453],
             [34,343,24]]])
print(x.shape)
print(x.ndim)

# Reshape
print("")
x = np.array([[0,1],
             [2,3],
             [4,5],
             [6,7]])
print(x.shape)

# Reshape unidimensional
x_reshape = x.reshape(8,1)
print(x_reshape)

# Reshape bidimensional
x_reshape = x.reshape(2,4)
print(x_reshape)

# Transpuesta de la matriz
x_transpose = np.transpose(x)
print(x_transpose)
print(x.T)