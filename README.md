
Inteligencia Artificial
======
Instituto Tecnológico de Costa Rica
Ingeniería en Computación
IC6200- Inteligencia Artificial

Grupo #8
------

- Fabio Mora Cubillo - 2013012801
- Sergio Moya Valerin - 2013015682
- Gabriel Venegas Castro - 2013115967

  

Contenidos
------

- Instalación
  > Dependencias
  > Modulo
- Uso
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
  > Arboles K-D y KNN

  

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
  Para ejecutar el predictor de votaciones se debe utilizar la siguiente estructura en cualquier programa de python:
```python
from tec.ic.ia.p1 import g08

g08.main()
```
Al terminar la ejecución el programa procederá a generar un archivo csv con el resultado en el directorio actual donde se ejecutó el programa. Además se muestra en la consola el nivel de precisión de la predicción.

Proyecto Corto 1 - Simulador Votos
======

  

Descripción
------

  

Consiste de un generador de un dataset de tamaño **n**. Este dataset generado será utilizado por el predictor de votaciones para predecir los resultados completos de las rondas 1 y 2 del proceso electoral del 2018, Costa Rica.

Se implementaron dos funciones principales, una para generar una muestra para todo el país y otro para generar una muestra por provincia, generar_muestra_pais y generar_muestra_provincia respectivamente.

Para ello se recolectó información del Tribunal Supremo de Elecciones respecto a los indicadores de cantón y las actas respectivas.

  

Implementación
------

  
  
  

Pruebas
------

En la carpeta test se encuentran archivos de prueba para los respectivos proyectos
Para el proyecto de simulación de votos están los siguientes archivos:
- **g08_pytest.py**: contiene las pruebas unitarias realizadas en *pytest*
- **g08_graficos.py**: contiene la generacion de graficos

  

> Los gráficos incluyen muestras de 1000, 10000 y 100000 para pais y tambien provincias.

  

![Distribución para una muestra de 1000 de Alajuela](test/pc1/graficos/Alajuela-1000.png)

  

Argumentos
------

**Simulador de votos**

> --indicadores <archivo.csv>
--actas <archivo.csv>

  

Ambos son evitables, puede especificarse uno, el otro, los dos o ninguno

  

Ejemplos
------

Para cargar las funciones principales del simulador se usa la siguiente línea:

```python
from tec.ic.ia.pc1.g08 import generar_muestra_pais, generar_muestra_provincia
```

La salida generada, en el caso de muestra de país, tendrá la siguiente forma, donde `...` son 31 columnas numericas con las caracteristicas del votante como por ejemplo escolaridad.

  

| Provincia | ... | Canton | Voto |
| -- | -- | -- | -- |
| Grecia | ... | Alajuela | Unidad Social Cristiana |
| Vazquez de Coronado | ...| San Jose | Voto blanco |
| Paraiso | ... | Cartago | Liberacion Nacional |
| San Rafael | ... | Heredia | Partido Accion Ciudadana |

  

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

  

Una precisión de entre 86% - 91%

  
  

Árboles K-D y KNN
------

  

**Implementación**

  

La implementación de árboles K-D y la búsqueda o recorrido de dichos árboles es una adaptación del código encontrado en la página ActiveState Code.

  

Las modificaciones realizadas a dicho código son:

  

- Datos de entrada, adaptados a los datos de prueba generados en el PC1
- Forma general, procesamiento correcto de los datos de entrada en el nuevo formato
- Cálculo del SqrtError, adaptado para ignorar el voto, tomando en cuenta sólo los indicadores
- Procesamiento de listas con vecinos cercanos
- Valores de retorno, ahora la función main retorna también la lista con

  

Para los árboles K-D se utilizó una implementación de clases. La clase KDTree representa el árbol K-D, y su generación se hace por recursión. Se realiza un ordenamiento de todos los elementos dependiendo del eje “axis” que representa la dimensión, o en términos más simples, la posición dentro de la lista de indicadores, y se hace una partición al medio. Se hace una llamada recursiva con los elementos separados a la izquierda de la media, y otra con los de la derecha.

  

La condición de parada se cumple cuando la cantidad de elementos en la hoja es menor o igual al valor máximo establecido.

  

La búsqueda de los N vecinos se implementó de manera recursiva

  
  

**Análisis de resultados**

  

Una precisión baja de alrededor de 19% - 24%

Se esperaba baja, pero no tanto. Se probó con diferentes valores para K y para el tamaño de las hojas, sin diferencias considerables

