import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles


N = 1000
LR = 0.0001

# Función de activación ReLU
def relu(x, derivate=False):
    if derivate:
        x[x<=0] = 0
        x[x>0] = 1
        return x
    else:    
        return np.maximum(0,x)

# Funcion de activación Sigmoide
def sigmoid(x, derivate = False):
    if derivate:
        return np.exp(-x)/(( np.exp(-x) +1)**2)
    else:    
        return 1 / (1 + np.exp(-x))

# Función de pérdida
def mse(y,y_hat,derivate=False):
    if derivate:
        return (y_hat - y)
    else:            
        return np.mean((y_hat - y)**2)

# Función de inicialización de parámetros
def initialize_parameters_dl(layers_dim):
    parameters = {}
    L = len(layers_dim)
    for l in range(0, L - 1):
        parameters[f'W{str(l+1)}'] = (np.random.rand(layers_dim[l],layers_dim[l+1]) * 2) - 1
        parameters[f'b{str(l+1)}'] = (np.random.rand(1,layers_dim[l+1]) * 2) - 1
    
    return parameters

gaussian_quantiles = make_gaussian_quantiles(mean=None,
                        cov=0.1,
                        n_samples=N,
                        n_features=2,
                        n_classes=2,
                        shuffle=True,
                        random_state=None
                        )

X, Y = gaussian_quantiles

Y = Y[:, np.newaxis]

# Problema a resolver con la red neuronal
plt.scatter(X[:,0], X[:,1], c=Y[:,0], s=40, cmap=plt.cm.Spectral)
# plt.show()

# layers_dims = [2,4,8,1]
# params = initialize_parameters_dl(layers_dims)
# print(params)

# print(params['W1'].shape)

# Operaciones de producto punto en numpy
# pp = np.matmul(X, params['W1'])
# print(pp.shape)
# print((X@params['W1']).shape)


# Función de entrenamiento
def train(x_data, lr, params, training = True):

    # Capas en Forward
    params['A0'] = x_data

    params['Z1'] = np.matmul(params['A0'],params['W1']) + params['b1']
    params['A1'] = relu(params['Z1'])
    
    params['Z2'] = np.matmul(params['A1'],params['W2']) + params['b2']
    params['A2'] = relu(params['Z2'])
       
    params['Z3'] = np.matmul(params['A2'],params['W3']) + params['b3']
    params['A3'] = sigmoid(params['Z3'])

    output = params['A3']

    # print(output.shape)

    if training:
        # Backpropagation
        params['dZ3'] =  mse(Y,output,True) * sigmoid(params['A3'],True)
        params['dW3'] = np.matmul(params['A2'].T,params['dZ3'])
        
        params['dZ2'] = np.matmul(params['dZ3'],params['W3'].T) * relu(params['A2'],True)
        params['dW2'] = np.matmul(params['A1'].T,params['dZ2'])
        
        params['dZ1'] = np.matmul(params['dZ2'],params['W2'].T) * relu(params['A1'],True)
        params['dW1'] = np.matmul(params['A0'].T,params['dZ1'])

        # Gradient Descent
        params['W3'] = params['W3'] - params['dW3'] * lr
        params['b3'] = params['b3'] - (np.mean(params['dZ3'],axis=0, keepdims=True)) * lr
        
        params['W2'] = params['W2'] - params['dW2'] * lr
        params['b2'] = params['b2'] - (np.mean(params['dZ2'],axis=0, keepdims=True)) * lr
        
        params['W1'] = params['W1'] -params['dW1'] * lr
        params['b1'] = params['b1'] - (np.mean(params['dZ1'],axis=0, keepdims=True)) * lr
    
    return output

# Entrenando la red
layers_dims = [2,4,8,1]
params = initialize_parameters_dl(layers_dims)
errors = []

for _ in range(100000):
    output = train(X,LR,params)
    if _ % 100 == 0:
        print(mse(Y,output))
        errors.append(mse(Y,output))
plt.plot(errors)
plt.show()

# Probando sobre datos nuevos
data_test = (np.random.rand(1000, 2) * 2) - 1
y = train(data_test,LR,params,training=False)

y = np.where(y >= 0.5, 1, 0)

plt.scatter(data_test[:,0], data_test[:,1], c=y[:,0] ,s=40, cmap=plt.cm.Spectral)
plt.show()


# Malla de visualización
_x0 = np.linspace(-1,1,50)
_x1 = np.linspace(-1,1,50)

_y = np.zeros((50,50))

for i0, x0 in enumerate(_x0):
    for i1, x1 in enumerate(_x1):
        _y[i0,i1] = train(np.array([[x0,x1]]),0.0001,params,training=False)

plt.pcolormesh(_x0,_x1,_y,cmap='coolwarm')
plt.show()