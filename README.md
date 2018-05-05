
Inteligencia Artificial
======
Instituto Tecnológico de Costa Rica

Ingeniería en Computación

IC6200 - Inteligencia Artificial

I Semestre,  2018

Grupo #8
------

- Fabio Mora Cubillo - 2013012801
- Sergio Moya Valerin - 2013015682
- Gabriel Venegas Castro - 2013115967

  

Contenidos
------

- Proyecto Corto 1 - Simulador Votos (pc1)
  > Descripcion
  > Implementacion
  > Pruebas
  > Argumentos
  > Ejemplos
- Proyecto 1 - Predicción Votaciones (p1)
  > Dataset
  > Modelos lineales
  > Redes neuronales
  > Árboles de decision
  > Árboles K-D y KNN
  > SVM
- Apéndice
  > Instalación
  > Uso

  


Proyecto Corto 1 - Simulador Votos
======

  

Descripción
------

  

Consiste de un generador de un dataset de tamaño **n**. Este dataset generado será utilizado por el predictor de votaciones para predecir los resultados completos de las rondas 1 y 2 del proceso electoral del 2018, Costa Rica.

Se implementaron dos funciones principales, una para generar una muestra para todo el país y otro para generar una muestra por provincia, generar_muestra_pais y generar_muestra_provincia respectivamente.

Para ello se recolectó información del Tribunal Supremo de Elecciones respecto a los indicadores de cantón y las actas respectivas.

  

Implementación
------
Para la implementación se creó un archivo .py que almacena los datos en formato csv para las actas y los indicadores por default. Igual son parametrizables pero cuando se cargan por default se utilizan estos archivos.

Primero se leen los parametros con argparse

```python
def parse_args():
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument(
        '--indicadores',
        nargs="+",
        default=['default'],
        help='Archivo csv. Ej: indicadores.csv')
    parser.add_argument(
        '--actas',
        nargs="+",
        default=['default'],
        help='Archivo csv. Ej: actas.csv')
    return parser.parse_args()
```

Luego se cargan los datos por default

```python
def load_data():
    args = parse_args()
    if args.indicadores[0] != 'default' and args.actas[0] != 'default':
        indicadores.INDICADORES = open(args.indicadores[0], 'r')
        actas.ACTAS_FINAL = open(args.actas[0], 'r')
    elif args.indicadores[0] == 'default' and args.actas[0] != 'default':
        actas.ACTAS_FINAL = open(args.actas[0], 'r')
    elif args.actas[0] == 'default' and args.indicadores[0] != 'default':
        indicadores.INDICADORES = open(args.indicadores[0], 'r')
```
Luego se formatea el csv a matriz de python de manera trivial. Para la función `generar_muestra_pais` lo que se hace es que se llama a la función `generar_muestra_provincia` para todas las provincias. Esta ultima función se trabaja de la siguiente manera:

 1. Se leen los archivos
```python
reader = csv.reader(StringIO(actas.ACTAS_FINAL))
acts = list(reader)
reader = csv.reader(StringIO(actas.ACTAS_FINAL2))
acts2 = list(reader)
reader = csv.reader(StringIO(indicadores.INDICADORES))
indicators = list(reader)
```
 2. Se obtiene el índice que separa los valores en el csv por provincia
```python
index = PROVINCIAS.index(nombre_provincia) * 32
```
 3. Una vez trabajados los índices para funcionar con el formato interno del archivo, se genera la población
```python
...
votes += get_vote(arr[:, j])
...
population += [[ # Demo-Geográficas
# Canton
(indicators[index][i]),
# Población total
float(indicators[index + 1][i]),
# Superficie
float(indicators[index + 2][i]),
...
# Porcentaje de hogares con jefatura
# compartida
get_att(indicators[index + 31][i]),
nombre_provincia, # Provincia
] + votes]
...
```
 4. Se retorna la población generada, en el caso de muestras de país, se concatena en el agregado de las poblaciones de distintas provincias
```python
population += generar_muestra_provincia(totals[i], PROVINCIAS[i],sample_type)
```

Pruebas
------

En la carpeta test se encuentran archivos de prueba para los respectivos proyectos
Para el proyecto de simulación de votos están los siguientes archivos:
- **g08_pytest.py**: contiene las pruebas unitarias realizadas en *pytest*
- **g08_graficos.py**: contiene la generacion de graficos

  

> Los gráficos incluyen muestras de 1000, 10000 y 100000 para pais y tambien provincias.

  

![Ejemplo de grafico: Distribución para una muestra de 1000 de Alajuela](test/pc1/graficos/Alajuela-1000.png)

  

Argumentos
------

**Simulador de votos**

| Parametro | Descripción | Obligatorio |
| -- | -- | -- |
| --indicadores | <archivo.csv> | No |
| --actas | <archivo.csv> | No |

**Predictor de votos**

| Parametro | Descripción | Obligatorio | Default |
| -- | -- | -- | -- |
| --provincia | nombre en mayuscula | No | 'PAIS' |
| --poblacion | cantidad de votantes | Si | N/A |
| --porcentaje-pruebas | define el tamano del set de pruebas | Si | N/A |
| |
| --regresion-logistica | flag | No | N/A |
| --l1 | cantidad de unidades por capa | No | N/A |
| --l2 | nombre de la funcion de activacion | No | N/A |
| |
| --red-neuronal | flag | No | N/A |
| --numero-capas | cantidad de capas | No | 1 |
| --unidades-por-capa | cantidad de unidades por capa | No | 60 |
| --funcion-activacion | nombre de la funcion de activacion | No | 'relu' |
| |
| --arbol | flag | No | N/A |
| --umbral-poda | porcentaje de poda del arbol | No | 0.1 |
| |
| --knn | flag | No | N/A |
| --k | k para el knn | No | 3 |

  

Ejemplos
------

Para cargar las funciones principales del simulador se usa la siguiente línea:

```python
from tec.ic.ia.pc1.g08 import generar_muestra_pais, generar_muestra_provincia
```

La salida generada, en el caso de muestra de país, tendrá la siguiente forma, donde `...` son 31 columnas numericas con las caracteristicas del votante como por ejemplo escolaridad.

  

| Provincia | ... | Canton | Voto 1ra ronda | Voto 2da ronda |
| -- | -- | -- | -- | -- |
| Grecia | ... | Alajuela | Unidad Social Cristiana | Partido Accion Ciudadana |
| Vazquez de Coronado | ...| San Jose | Voto blanco | Restauracion Nacional |
| Paraiso | ... | Cartago | Liberacion Nacional | Voto blanco |
| San Rafael | ... | Heredia | Partido Accion Ciudadana | Partido Accion Ciudadana |

  

Evidentemente el comportamiento es el mismo para generar muestra por provincia con la diferencia de que solo genera votantes en cantones de la provincia especificada.

  

Proyecto 1 - Predicción Votaciones
======

  

Dataset
------

Se utiliza el simulador de votos para generar el dataset que se utilizara en la ejecución de todos los modelos como se explica anteriormente. Se importan de la manera especificada y luego se trabaja los datos tomando en cuenta el porcentaje de población que se utilizara.

  

Este proyecto cuenta con la funcion shaped_data_no_bin que devuelve el dataset formateado para funcionar con el voto siendo un string que contiene el nombre del partido, además también cuenta con otra funcion llamada shaped_data que funciona con el voto siendo una lista conteniendo un valor para todos los partidos, siendo este 1 para el partido por el cual se votó y 0 para el resto.

  

Modelos lineales
------

  

**Implementación**

Para implementar la regresión logística se trabajan los siguientes parámetros l1 y l2 para indicar el los niveles de regularización.

  

Se utilizó de backend Tensorflow con Keras (API de python que corre sobre tensorflow).

  

De la librería sklearn se utilizó de preprocesamiento OneHotEncoder para entrenar el modelo. Luego la funcion sigmoid para la predicción y finalmente se le calculan los costos en relación a la entropía.

  

```python
oneHot = OneHotEncoder()
oneHot.fit(X)
...
prediction = tf.nn.sigmoid(Z)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z, labels = y))
```

  

**Análisis de resultados**

 Resultado con L1
```
Regresion con l1 True , l2 False
REGRESION_LOGISTICA
   - Error de entrenamiento:  47.21666666666667
   - Error de pruebas:  0.149284680684
```
 Resultado con L2
```
Regresion con l1 False , l2 True
REGRESION_LOGISTICA
   - Error de entrenamiento:  57.666666666666664
   - Error de pruebas:  0.149284680684
```
  Como se puede apreciar, la potencia del modelo con L1 es de 53% para el entrenamiento y 86% para las pruebas. De igual manera con L2 el entrenamiento tuvo 43% de potencia y las pruebas 86%.
  

Redes neuronales
------

**Implementación**

Para implementar las redes neuronales se aceptan los siguientes parámetros:

  

| Parametro | Valor valido |
| --- | --- |
| numero-capas | entero mayor a 0 |
| unidades-por-capa | numero mayor a 0 |
| funcion-activacion | relu/sigmoid/softmax |

  

Para estos modelos también se utilizó Keras con Tensorflow

  

Se creo una funcion que crea el modelo de acuerdo a los parámetros que pasa el usuario utilizando el modelo Secuencial de Keras de la siguiente manera:

```python
def baseline_model(y_len, hidden_layer_amount, hidden_unit_amount, activation_fun):
    model = Sequential()
    model.add(Dense(66, input_dim=33, activation=activation_fun))
    while hidden_layer_amount > 0:
        model.add(Dense(hidden_unit_amount, activation=activation_fun))
        hidden_layer_amount -= 1
    model.add(Dense(y_len, activation='softmax'))
    model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
    return model
```

Una vez el programa crea el modelo, lo ejecuta en un KerasClassifier, que se encarga de entrenar el modelo para obtener un estimador, el cual se utilizara en el kfold cross validation. El kfold es una técnica importante para estimar la precisión del modelo. Dicha técnica se puede apreciar en el siguiente código:

```python
estimator = KerasClassifier(
        build_fn=baseline_model,
        y_len=len(dummy_y[0]),
        hidden_layer_amount=hidden_layer_amount,
        hidden_unit_amount=hidden_unit_amount,
        activation_fun=activation_fun,
        epochs=100,
        batch_size=100,
        verbose=2)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
```

**Análisis de resultados**

Inicialmente se daban casos en los que se predecian para la segunda ronda resultados con partidos politicos que no correspondian, por lo que se determino que algo malo sucedia con el modelo. Luego de algunos arreglos se logro que se prediga bien.
 
Con funcion de activacion `'sigmoid'`
```
Red neuronal con 1 capas, 60 unidades por capa y  como funcion de activacion
RED_NEURONAL
   - Error de entrenamiento:  48.19597989949748
   - Error de pruebas:  51.0833333333
``` 
Con funcion de activacion `'relu'`
```
Red neuronal con 1 capas, 60 unidades por capa y relu como funcion de activacion
RED_NEURONAL
   - Error de entrenamiento:  41.19430485762144
   - Error de pruebas:  50.6675881112
```
Con funcion de activacion `'softmax'`
```
Red neuronal con 1 capas, 60 unidades por capa y softmax como funcion de activacion
RED_NEURONAL
   - Error de entrenamiento:  39.904522613065325
   - Error de pruebas:  48.8156388067
```
Pese a ser la misma cantidad de capas y de unidades por cada una, es observable una diferencia importante en la potencia, de acuerdo a la funcion de activacion que utilicen.

Árboles de decisión
------

**Implementación**

**Análisis de resultados**


Corrida con umbral de poda default (10%)
```
Arbol con umbral de poda 0.1
DECISION_TREE
   - Error de entrenamiento:  0.585427135678392
   - Error de pruebas:  0.9583333333333334
```
Corrida con umbral de poda al 80%
```
Arbol con umbral de poda 0.8
DECISION_TREE
   - Error de entrenamiento:  0.7361809045226131
   - Error de pruebas:  0.8816666666666667
```
Es importante ver como entre mas umbral de poda mas baja la potencia del entrenamiento y suba la potencia de las pruebas.

Árboles K-D y KNN
------

  

**Implementación**

  

La implementación de árboles K-D y la búsqueda o recorrido de dichos árboles es una adaptación del código encontrado en la página ActiveState Code.

  

Las modificaciones realizadas a dicho código son:

  

- Datos de entrada, adaptados a los datos de prueba generados en el PC1
- Forma general, procesamiento correcto de los datos de entrada en el nuevo formato compatible con el generador de poblaciones
- Cálculo del SqrtError, adaptado para ignorar el voto, tomando en cuenta sólo los indicadores (Sin embargo, el voto de primera ronda es considerado un indicador cuand así se requiere )
- Procesamiento de listas con vecinos cercanos
- Valores de retorno conformes con lo necesario para la obtención de información de salida

  

Para los árboles K-D se utilizó una implementación de clases. La clase KDTree representa el árbol K-D, y su generación se hace por recursión. Se realiza un ordenamiento de todos los elementos dependiendo del eje “axis” que representa la dimensión, o en términos más simples, la posición dentro de la lista de indicadores, y se hace una partición al medio. Se hace una llamada recursiva con los elementos separados a la izquierda de la media, y otra con los de la derecha.

La estructura Node es la que contiene el formato utilizado para la creación de nodos del árbol. 
Contiene 
- point: Punto medio (mediana) del eje que separa a los hijos izquierdo y derecho del nodo
- axis: El eje que funciona como critero de comparación para agregar nodos
- label: La etiqueta asignada al nodo
- left: El hijo izquierdo
- right: El hijo derecho

```python
Node = collections.namedtuple("Node", 'point axis label left right')

#Generación del árbol recursivamente
def build_tree(objects, axis=0):

            if not objects:
                return None
            objects.sort(key=lambda element: element[axis])
            median_idx = len(objects) // 2
            median_point= objects[median_idx][0:-1]
            median_label = objects[median_idx][-1]
            next_axis = (axis + 1) % k
            return Node(median_point, axis, median_label,
                        build_tree(objects[:median_idx], next_axis),
                        build_tree(objects[median_idx + 1:], next_axis))

        self.root = build_tree(list(objects))
```  



La condición de parada se cumple cuando la cantidad de elementos en la hoja es menor o igual al valor máximo establecido.

  

La búsqueda de los N vecinos se implementó de manera recursiva

Para realizar Cross Validation (K-Fold) se toma el set de datos completo, se le separa el porcentaje reservado para pruebas (por defecto un 20%) y al restante (80%) se le separa en una cantidad de partes predeterminada (10 por defecto). Cada una de esas partes actúa como set de prueba mientras las demás actúan como set de entrenamiento. Posteriormente se selecciona el árbol que obtuvo las mejores predicciones y se realizan las pruebas finales.

Para la salida del modelo, se retorna un diccionario con: 
|  Predicción 1ra ronda  | Predicción 2da ronda sin 1ra | Predicción 2da ronda con 1ra | Es entrenamiento | Error de entrenamiento | Error de pruebas
| -- | -- | -- | -- |  -- | -- |
| Voto  | Voto | Voto | True/False | Valor | Valor
| Voto  | Voto | Voto | True/False |
| ..  | .. | .. | .. | 
Este diccionario contiene todo lo necesario para construir el archivo CSV requerido para el proyecto.

**Análisis de resultados**

Con los árboles generados mediante cross validation, los mejores árboles obtuvieron:
- Precisión baja de alrededor de 19% - 24% para las predicciones de primera ronda.
- Precisión de alrededor de 55% - 60% para las predicciones de segunda ronda sin tomar en cuenta los votos de la primera ronda. 
- Precisión de alrededor de 55% - 60% para las predicciones de segunda ronda tomando en cuenta los votos de primera ronda

No hay una diferencia considerable entre la precisión de las predicciones de segunda ronda con y sin votos de primera ronda. Se intentó modificar el orden de los indicadores poblacionales para buscar una mejoría en la precisión sin éxito. Se especula que esta situación se deba a la naturaleza de los datos, o al formato de los mismos al ser ingresados a la generación del árbol. 

Para las pruebas con la porción del dataset considerada "datos nuevos", 20% de la muestra original, la precisión bajó con respecto a las pruebas realizadas mediante cross validation al dataset de entrenamiento. En promedio, el error aumentó entre 5% y 15% para las predicciones de todas las rondas.

En síntesis, se esperaba una baja precisión, pero no tan baja.
Se probó con diferentes valores para K y para el tamaño de las hojas, sin diferencias notables.

Corrida con 6 como valor de k.
```
KNN con k =  6
KNN
   - Error de entrenamiento:  0.71
   - Error de pruebas:  0.633333333333334
```
Corrida con 4 como valor de k.
```
KNN con k =  4
KNN
   - Error de entrenamiento:  0.647141261864879965242881072027
   - Error de pruebas:  0.69733333333333333
```
El comportamiento muestra resultados diferentes dependiendo del valor de k. Siendo k 6 se obtuvo mas error de entrenamiento que siendo k 4. De manera opuesta, con k = 6 el error de pruebas fue menor que con k = 4.

SVM
------

**Implementación**

Para implementar el SVM se escogio un modelo lineal.

Primero se divide el dataset:
```python
x_train, x_test, y_train, y_test = non_shuffling_train_test_split(X1, Y1, test_percentage/100)
```

Luego se crea el modelo utillizando `sklearn.svm`
```python
model = LinearSVC()
model.fit(x_train, y_train.ravel())
```

Para predecir se utiliza:
```python
predictions = model.predict(x_train)
```

**Análisis de resultados**

Resultado de una corrida:
```
SVM
   - Error de entrenamiento:  0.551088777219
   - Error de pruebas:  0.482873851295
```
El resultado de otra corrida:
```
SVM
   - Error de entrenamiento:  0.529313232831
   - Error de pruebas:  0.452798663325
```
Los resultados evidencian que el modelo de SVM anda entre un 44-49% de potencia en el entrenamiento y 51-55% en las pruebas.

Apéndice
======
Instalación
------

  

**Dependencias**

  

tensorflow

```bash
python -m pip install tensorflow
```

sklearn

```bash
python -m pip install sklearn
```

pandas

```bash
python -m pip install pandas
```

keras

```bash
python -m pip install keras
```

pptree

```bash
python -m pip install pptree
```

  

**Modulo (Opcion #1):**

```bash
python setup.py sdist
python -m pip install dist/tec-2.X.tar.gz
```

**Modulo (Opcion #2):**

```bash
python setup.py install
```

  
Uso
------
**Estructura de codigo**

Para ejecutar el predictor de votaciones se debe utilizar la siguiente estructura en cualquier programa de python:
```python
# importar el modulo
from tec.ic.ia.p1 import g08
# obtener la prediccion
g08.run_prediction()
```
Al terminar la ejecución el programa procederá a generar un archivo csv con el resultado en el directorio actual donde se ejecutó el programa. Además se muestra en la consola el nivel de precisión de la predicción.

**Ejemplo**

Considerando que un archivo python `ejemplo.py` tenga la estructura anterior y que lo que queremos es obtener la prediccion utilizando un modelo de red neuronal, lo llamamos de la siguiente manera:

```bash
python ejemplo.py --poblacion 1003 --porcentaje-pruebas 20 --red-neuronal --unidades-por-capa 60
```
Lo que nos va a generar un archivo llamado `red_neuronal_output.csv` y una salida en la consola como la siguiente:
```bash
RED_NEURONAL
    - Error de entrenamiento: 0.4582124
    - Error de pruebas: 0.4012345
```
