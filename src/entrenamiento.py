from descenso import gradient_descent, plotChart, calcular_costo
from data_management import clean_data
import pandas as pd
import numpy as np

# Separar conjunto de entrenamiento y de prueba

def main():

    # Limpiar data
    clean_data('../doc/CarDekho.csv', 2)

    # Cargar data
    data = pd.read_csv('../doc/CarDekho_clean.csv')
    #print("Datos cargados")
    #print(data)

    # Obtener columna Price para y
    y = data['Price']

    # Filtrar columnas para X
    X = data.drop('Price', axis=1)

    print("Datos filtrados")
    print(f"X: {X}")
    print(f"y: {y}")

    # Se normalizan las caracteristicas X
    # para evitar que las características con valores más grandes 
    # tengan un impacto desproporcionado en el proceso de aprendizaje

    ## Normalizacion de la data con valores numericos altos
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