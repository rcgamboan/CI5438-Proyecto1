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

def main():

    # Import data
    data = pd.read_csv('doc/house_practice.csv')

    # Extract data into X and y
    X = data[['Size', 'Bedrooms']]
    y = data['Price']

    # Se normalizan las caracteristicas X
    # para evitar que las características con valores más grandes 
    # tengan un impacto desproporcionado en el proceso de aprendizaje

    # Si las características tienen valores muy diferentes, 
    # el gradiente puede ser muy sensible a las características con valores más grandes. 
    # Esto puede provocar que el modelo se ajuste demasiado a estas características y 
    # que se vea afectado por el ruido.
    X = (X - X.mean()) / X.std()

    # Se le agrega otra columna a X con 1 para permitir el descenso de gradiente vectorizado
    # En lugar de calcular el gradiente para cada característica individualmente, 
    # el descenso de gradiente vectorizado calcula el gradiente para todas las características a la vez.
    X = np.c_[np.ones(X.shape[0]), X] 


    # Tasa de aprendizaje y numero de iteraciones
    # Si la tasa de aprendizaje es demasiado alta, 
    # el modelo puede oscilar entre diferentes mínimos locales y no converjer a un minimo global. 
    # Si la tasa de aprendizaje es demasiado baja, el modelo puede tardar mucho tiempo en converger. 
    # En algunos casos, el modelo puede no converger en absoluto.
    alpha = 0.0005
    iterations = 10000

    # Inicializar vector theta
    theta = np.zeros(X.shape[1])

    # Ejecutar descenso de gradiente
    # al ejecutar el algoritmo, se obtiene una ecuacion de la forma
    # y = theta_0 + theta_1 * x_1 + theta_2 * x_2
    # donde:
    #   y es lo que se quiere predecir
    #   theta_0 es el término independiente
    #   theta_1 es el coeficiente de la caracteristica x1
    #   theta_2 es el coeficiente de la característica x2
    theta, lista_costos, iters = gradient_descent(X, y, theta, alpha, iterations)

    # Mostrar grafica de costo vs iteraciones
    plotChart(iters, lista_costos[0:iters])

    costo_final, _ = calcular_costo(X, y, theta)

    print(f"Costo final: {costo_final}\ntheta final: {theta}\n")

if __name__ == "__main__":
    main()