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
    
# Función de activación ReLU
def relu(x, derivate=False):
    if derivate:
        return np.where(x > 0, 1, 0)
    else:
        return np.maximum(0, x)

def tanh(x, derivate=False):
    if derivate:
        return 1 - np.tanh(x)**2
    else:
        return np.tanh(x)

def grafica(n, x, f, name):
    plt.figure(n)
    plt.plot(x, f)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Función {name}')
    plt.show()
    
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
grafica(1, x, sigmoid(x), "Sigmoid")

# # Crear la segunda figura y gráfico
grafica(2, x, step(x), "Step")

# Crear la tercera figura y grafico
grafica(3, x, tanh(x), "Tanh")

# Ejemplo de uso ReLu
x = np.array([-2, -1, 0, 1, 2])

# Crear la cuarta figura y grafico
grafica(4, x, relu(x), "ReLu")


# Ejemplo de función de pérdida
prediction = np.array([0.9,0.5,0.2,0.0])
real =  np.array([0,0,1,1])

mse_result = mse(real, prediction)
print(mse_result)


if __name__ == '__main__':
    print('Worked!!')