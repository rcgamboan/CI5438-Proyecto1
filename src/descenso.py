import numpy as np
import pandas as pd
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