import numpy as np
import matplotlib.pyplot as plt
import random

def gradient_descent(X, y, theta, alpha, iteraciones):
  """
  Implementa el algoritmo de descenso de gradiente para regresión lineal multivariada.

  Args:
    X: Matriz de datos de entrada.
    y: Matriz de datos de salida.
    theta: Vector de parámetros iniciales.
    alpha: Tasa de aprendizaje.
    iteraciones: Número de iteraciones.

  Returns:
    Vector de parámetros optimizados.
  """
  costos = []

  for iter in range(iteraciones):

    h = np.dot(X, theta)
    loss = (h - y)**2
    #loss = np.sum((h - y)**2)

    #gradient = np.dot(X.T, loss) / X.shape[0]

    gradient = X.T.dot(h - y) * 2/X.shape[0]

    theta = theta - alpha * gradient

    cost_value = 1/(2*len(y))*(loss) # Calcular costo de la iteracion
    
    total = np.sum(cost_value)
    
    costos.append(total)
  
  plt.plot(np.arange(1,iteraciones),costos[1:], color = 'red')
  plt.title('Costo vs # de iteracion')
  plt.xlabel('Iteracion #')
  plt.ylabel('Costo')
  plt.show()
  
  return theta


def generar_datos(cant_puntos, bias, variance, cant_variables):
  x = np.zeros(shape=(cant_puntos, cant_variables))
  y = np.zeros(shape=cant_puntos)
  # basically a straight line
  for i in range(0, cant_puntos):
    for j in range(cant_variables):
      x[i][j] = random.uniform(0, 1) * variance + i
    y[i] = (i + bias) + random.uniform(0, 1) * variance
  return x, y

def main():
  # Datos de entrenamiento
  # X = np.array([[1, 2], [3, 4], [5, 6]])
  # y = np.array([3, 7, 11])

  x, y = generar_datos(5, 25, 10,2)

  # Parámetros iniciales
  hipotesis_inicial = np.ones(x.shape[1])

  # Tasa de aprendizaje
  alpha = 0.0001

  # Número de iteraciones
  iteraciones = 1000

  # Entrenamiento del modelo
  hipotesis = gradient_descent(x, y, hipotesis_inicial, alpha, iteraciones)

  #print(f"X: {x}")
  #print(f"y: {y}")
  print(f"hipotesis: {hipotesis}")

  """
  # Initialize layout
  fig, ax = plt.subplots(figsize = (9, 9))

  # Add scatterplot
  ax.scatter(x, y, s=60, alpha=0.7, edgecolors="k")

  # Fit linear regression via least squares with numpy.polyfit
  # It returns an slope (b) and intercept (a)
  # deg=1 means linear fit (i.e. polynomial of degree 1)
  b = hipotesis[0]
  a = hipotesis[1]

  # Create sequence of 100 numbers from 0 to 100 
  xseq = np.linspace(0, 10, num=100)

  # Plot regression line
  ax.plot(xseq, a + b * xseq, color="k", lw=2.5)

  # Set title and labels
  ax.set_title("Regresión lineal")
  ax.set_xlabel("X")
  ax.set_ylabel("y")
  ax.grid(True)
  ax.show()"""


if __name__ == "__main__":
  main()
