from descenso import gradient_descent, plotChart, calcular_costo
from data_management import clean_data
import pandas as pd
import numpy as np

# Separar conjunto de entrenamiento y de prueba

def main():

    # Cargar y limpiar data
    data, precios_reales = clean_data('../doc/CarDekho.csv', 2)

    # Separar data de entrenamiento y de prueba
    # se utiliza un 80% para entrenamiento y 20% para prueba
    sep = int(data.shape[0]*0.8)
    train = data.iloc[:sep,:]
    test = data.iloc[sep:,:]

    # Separando los precios reales de los conjuntos de entrenamiento y prueba
    max_price = precios_reales.max()
    min_price = precios_reales.min()
    precios_reales_train = precios_reales.iloc[:sep]
    precios_reales_test = precios_reales.iloc[sep:]

    # Obtener columna Price para y
    y = train['Price']

    # Filtrar columnas para X
    X = train.drop('Price', axis=1)

    print("Datos entrenamiento cargados")
    print(f"X: {X}")
    print(f"y: {y}")

    # Se normalizan las caracteristicas X
    # para evitar que las características con valores más grandes 
    # tengan un impacto desproporcionado en el proceso de aprendizaje.
    # Si las características tienen valores muy diferentes, 
    # el gradiente puede ser muy sensible a las características con valores más grandes. 
    # Esto puede provocar que el modelo se ajuste demasiado a estas características y 
    # que se vea afectado por el ruido.
    #X = (X - X.min()) / (X.max() - X.min())
    #print(f"X normalizado: {X}")

    # Se le agrega otra columna a X con 1 para permitir el descenso de gradiente vectorizado
    # En lugar de calcular el gradiente para cada característica individualmente, 
    # el descenso de gradiente vectorizado calcula el gradiente para todas las características a la vez.
    X = np.c_[np.ones(X.shape[0]), X]


    # Tasa de aprendizaje y numero de iteraciones
    # Si la tasa de aprendizaje es demasiado alta, 
    # el modelo puede oscilar entre diferentes mínimos locales y no converger a un minimo global. 
    # Si la tasa de aprendizaje es demasiado baja, el modelo puede tardar mucho tiempo en converger. 
    # En algunos casos, el modelo puede no converger en absoluto.
    alpha = 0.00005
    iterations = 100000

    # Inicializar vector theta
    theta = np.zeros(X.shape[1])

    # Ejecutar descenso de gradiente
    # al ejecutar el algoritmo, se obtiene una ecuacion de la forma
    # y = theta_0 + theta_1 * x_1 + theta_2 * x_2 + ... + theta_n * x_n
    # donde:
    #   y es lo que se quiere predecir
    #   theta_0 es el término independiente
    #   theta_1 es el coeficiente de la caracteristica x1
    #   theta_2 es el coeficiente de la característica x2
    theta, lista_costos, iters = gradient_descent(X, y, theta, alpha, iterations)

    print(f"iteraciones: {iters}")

    # Mostrar grafica de costo vs iteraciones
    plotChart(iters, lista_costos[0:iters])

    costo_final, _ = calcular_costo(X, y, theta)
    print(f"Costo final: {costo_final}\ntheta final: {theta}\n")

    # Predecir precios de la data de prueba
    test_price = test['Price']
    test = test.drop('Price', axis=1)
    test = np.c_[np.ones(test.shape[0]), test]
    y_pred = theta.dot(test.T)

    # Calcular error de predicción
    errores = y_pred - test_price

    # Calcular RMSE
    rmse = np.sqrt(np.mean(errores**2))

    # Imprimir RMSE
    print(f"RMSE: {rmse}")

    # Desnormalizar precios
    for price in range(len(y_pred)):
        y_pred[price] = (y_pred[price] * (max_price - min_price)) + min_price
    print("Predicciones:")
    for pred in range(len(y_pred)):
        print(f"Prediccion: {y_pred[pred]} Precio real: {precios_reales_test.iloc[pred]}")




if __name__ == "__main__":
    main()