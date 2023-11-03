import numpy as np
import random
import matplotlib.pyplot as plt

def calcular_costo(X, y, theta):
    m = y.size
    hipotesis = np.dot(X, theta.T)
    error = hipotesis - y
    costo = np.sum((hipotesis-y)**2) / m
    
    return costo, error

def gradient_descent(X, y, theta, alpha, iters, tolerancia = 1e-6):
    """
    Implementa el algoritmo de descenso de gradiente para regresión lineal multivariada.

    Args:
        X: Matriz con los datos de las caracteristicas.
        y: Matriz de datos de salida.
        theta: Vector de parámetros iniciales.
        alpha: Tasa de aprendizaje.
        iteraciones: Número de iteraciones.
        tolerancia: Tolerancia para la convergencia.

    Returns:
        theta: Vector theta con los pesos de la hipotesis.
        cost_array: Arreglo con los costos de cada iteracion.
        iters: Numero de iteraciones realizadas antes de finalizar el algoritmo.
    """
    
    cost_array = np.zeros(iters)
    m = y.size
    costo_ant = None

    for i in range(iters):

        costo_act, error = calcular_costo(X, y, theta)
        
        # Convergencia
        if costo_ant and abs(costo_ant-costo_act)<=tolerancia:
            print(f"Convergencia en iteracion {i}\n")
            iters = i
            break

        costo_ant = costo_act
        grad = np.dot(X.T, error)
        theta -= (alpha * (1/m) * grad)
        cost_array[i] = costo_act

    return theta, cost_array, iters

def plotChart(iterations, cost_num):
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), cost_num, 'r')
    ax.set_xlabel('Iteracion #')
    ax.set_ylabel('Costo')
    ax.set_title('Costo vs # Iteracion')
    plt.style.use('fivethirtyeight')
    plt.show()

def generar_data(cant_puntos,w0,w1,w2):
    x = np.zeros(shape=(cant_puntos,2))
    y = np.zeros(shape=cant_puntos)

    for i in range(cant_puntos):
        x[i][0] = random.randint(1,100)
        x[i][1] = random.randint(1,100)
        y[i] = w0 + w1*x[i][0] + w2*x[i][1]
    
    return x, y

def main():

    X_init, y = generar_data(1000, -10, 2,5)
    X = np.c_[np.ones(X_init.shape[0]), X_init]

    alpha = 0.00000005
    iterations = 100000
    theta = np.zeros(X.shape[1])
    
    theta, lista_costos, iters = gradient_descent(X, y, theta, alpha, iterations)
    
    # Se redondean los valores obtenidos 
    w0 = round(theta[0],2)
    w1 = round(theta[1],2)
    w2 = round(theta[2],2)

    print("\nfuncion obtenida con el descenso de gradiente")
    print(f"f(x1,x2) = {w1} x1 + {w2} x2 + {w0}\n")
    
    plotChart(iters, lista_costos[0:iters])

    X_init = np.c_[np.ones(X_init.shape[0]), X_init]
    y_pred = theta.dot(X_init.T)

    # Calcular error de predicción
    errores = y_pred - y

    # Calcular RMSE
    rmse = np.sqrt(np.mean(errores**2))

    # Imprimir RMSE
    print(f"Error en la predicción\nRMSE: {rmse}")


if __name__ == "__main__":
    main()