# CI5438-Proyecto1

El objetivo de este proyecto es implementar el algoritmo de descenso de gradiente con el error cuadrático como función de pérdida, tal y como fue visto en clases, para regresión lineal multivariada. Luego probaremos el algoritmo con un caso controlado para verificar su correcto funcionamiento, y finalmente lo aplicaremos sobre un conjunto de datos que deberá ser pre-procesado apropiadamente.

La elección de lenguaje de programación es libre. Más allá de la implementación del algoritmo, la cuál es obligatoria, puede usar cualquier libreria o herramienta que considere necesaria para facilitar su trabajo.

## Parte 1: Implementación

Implemente el algoritmo de descenso de gradiente para regresión lineal multivariada, usando la función de pérdida cuadrática $L^2(y,\hat{y}) = (y - \hat{y})^2$, tal como fue estudiado en la clase de regresión lineal. Agregar un factor de regularización es opcional. Utilice como condición de convergencia (se sugiere las haga configurables, para facilidad a la hora de hacer pruebas):
* Una cantidad máxima de iteraciones.
* Que la función de pérdida sea menor que un $\epsilon$ pequeño.

Para validar que su implementación funciona correctamente, cree una función lineal de la forma $f(x) = w_1 x_1 + w_2 x_2 + w_0$, seleccionando valores no nulos para los pesos $w_0$, $w_1$, y $w_2$. Usando $f(x)$, genere 1000 puntos aleatorios, que usará como datos de entrenamiento para un modelo lineal. Verifique que logra la convergencia por la segunda condición propuesta ($\epsilon$ pequeño), y que los coeficientes obtenidos se corresponden **aproximadamente** a los escogidos.

Tome en cuenta que el descenso de gradiente es un método numérico, y tal vez no consiga resultados perfectos, dependiendo de los valores escogidos. Tome en cuenta también que pudiera necesitar de un epsilon bastante pequeño y una cantidad de iteraciones grande para alcanzar la convergencia, dependiendo de cómo escoja la función lineal y los puntos a usar como conjunto de entrenamiento (considere cómo se calcula cada actualización y el funcionamiento del algoritmo para entender cómo funciona esta relación).

Haga que su programa grafique, para el caso particular donde consiguió convergencia, la curva de entrenamiento (el número de iteración como eje x y el error en la iteración actual en el eje y).

## Parte 2: Preprocesamiento de datos

En el repositorio se incluye un conjunto de datos `CarDekho.csv`, el cuál contiene los siguientes datos:
* Make (fabricante)
* Model (modelo)
* Price (precio de venta)
* Year (año)
* Kilometer (kilometraje)
* Fuel Type (tipo de combustible)
* Transmission (tipo de transmisión)
* Location (ubicación)
* Color (color)
* Owner (nro. de dueño actual)
* Seller Type (tipo de vendedor)
* Engine (motor)
* Max Power (potencia máxima)
* Max Torque (par máximo del motor)
* Drivetrain (tren motriz)
* Length (largo)
* Width (ancho)
* Height (alto)
* Seating Capacity (cantidad de asientos)
* Fuel Tank Capacity (capacidad del tanque de combustible)

Se quiere que usted haga un modelo de regresión lineal para predecir el **precio de venta** (Price) de un automóvil, dado cierta combinación de sus características. Se sugiere que use como atributos de entrada:
* Make
* Year
* Kilometer
* Fuel Type
* Transmission
* Owner
* Seating Capacity
* Fuel Tank Capacity

Pero usted es libre de escoger otros atributos para su modelo. Discuta su decisión apropiadamente si decide hacerlo.

Antes de usar estos datos, debe hacer operaciones de preprocesamiento sobre ellos, la primera siendo separar solo los atributos que usará para su modelo.

### Manejo de valores faltantes

Algunos elementos del conjunto de datos tienen atributos faltantes. Puede lidiar con esto de distintas maneras:
* Remover los ejemplos totalmente.
* Reemplazar los valores faltantes por la mediana del atributo.
* Reemplazar los valores faltantes por el valor más común del atributo.

Es posible que quiera lidiar con esto de manera distinta dependiendo del atributo faltante. Refleje su decisión apropiadamente en su informe.

### Manejo de valores categóricos

Usted notará que varios atributos del conjunto no tienen valores continúos, sino que son asignados a una categoría, lo cuál impide que sean usados tal cuál para su modelo. La manera en la que manejará esto es través de **variables dummy**. La técnica funciona de la manera siguiente
1) Separar cada atributo categórico con $n$ valores posibles en $n$ atributos distintos.
2) Para cada ejemplo, si su atributo pertenecía a la categoría que corresponde a la variable, le asignaremos 1. De otra forma, le asignaremos 0.

De esta forma, para un atributo con $n$ clases terminaremos con $n$ variables binarias. Considere que ciertas variables numéricas, en realidad podrían tener más sentido al considerarse como categóricas. Discuta en su informe su elección de variables para codificar.

### Normalización

Es posible que sea más fácil conseguir convergencia si efectúa **normalización** sobre los datos que presentan valores muy grandes en relación a los otros. Una manera de hacerlo es aplicando la siguiente operación sobre el atributo $X$ que se quiere normalizar:

$$
X' = \frac{X - X_{min}}{X_{max} - X_{min}}
$$

La cuál restringirá el atributo a valores entre $0$ y $1$.

**Cualquier otra operación de preprocesamiento de datos que efectúe, debe quedar reflejada en su informe.**

## Parte 3: Entrenamiento del modelo

Ya teniendo un conjunto de datos manejable para ajustar un modelo lineal, debe aplicar un proceso de **validación cruzada**. Para ello, debe separar su conjunto de datos en conjuntos de **entrenamiento** y **prueba**, para lo cuál se sugiere una separación de 80% y 20% respectivamente. Puede elegir seleccionar también un conjunto de **validación**, pero no es obligatorio. Note que **no** es obligatorio que use el conjunto de datos por completo para sus experimentos.

Para el proceso de entrenamiento, asegúrese de probar con distintos valores para la tasa de entrenamiento. Grafique las curvas de aprendizaje para los experimentos realizados.

Finalmente, seleccione una hipótesis, justificando su elección.

## Entrega

Para la entrega del proyecto, haga **fork** de este repositorio. Su repositorio deberá contener todo el código usado para el proyecto, las gráficas que fueron generadas durante el entrenamiento, y un informe discutiendo los detalles de su implementación, su proceso de preprocesamiento y su proceso de entrenamiento.
