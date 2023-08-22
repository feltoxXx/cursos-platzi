import numpy as np
import matplotlib.pyplot as plt


# Funcion de activación Sigmoide
def sigmoid(a, derivate = False):
    if derivate:
        return np.exp(-a)/(( np.exp(-a) +1)**2)
    else:    
        return 1 / (1 + np.exp(-a))

# Funcion de activación Step
def step(x, derivate = False):    
    return np.piecewise(x,[x<0.0,x>=0.0],[0,1])

# Función de pérdida
def mse(y,y_hat,derivate=False):
    if derivate:
        return (y_hat - y)
    else:            
        return np.mean((y_hat - y)**2)
    
# creamos valores espaciados para el eje x
x = np.linspace(10,-10,100)

# Crear un gráfico con las dos funciones
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, step(x), label='Step')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Funciones Sigmoid y Step')
plt.show()


# # Crear la primera figura y gráfico
# plt.figure(1)
# plt.plot(x, sigmoid(x))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Función Sigmoid')
# plt.show()

# # Crear la segunda figura y gráfico
# plt.figure(2)
# plt.plot(x, step(x), color='orange')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Función Step')
# plt.show()

prediction = np.array([0.9,0.5,0.2,0.0])
real =  np.array([0,0,1,1])

mse_result = mse(real, prediction)
print(mse_result)