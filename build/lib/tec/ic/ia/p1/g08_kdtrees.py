import collections
import itertools
import math
import random
import numpy
from tec.ic.ia.p1 import g08_data
from tec.ic.ia.pc1 import g08

def getParsedData(n):

    data = g08_data.shaped_data_no_bin(n).tolist()
    data = [[float(i) for i in person] for person in data]
    return data


def genDataSet(n, m):
    dataSet = []
    dataSet = [list(random.randint(0,1) for x in range(m)) for y in range(n)]
    for i in dataSet:
        if i[3]%2 == 0:
            i[-1] = 'PAC'
        else:
            i[-1] = 'RN'


    print(dataSet[0])

    return dataSet


def square_distance(a, b):
    square = 0
    a = a[0:-1]
    b = b[0:-1]
    for elemX, elemY in zip(a, b):
        dist = elemX - elemY
        square += dist * dist
    return square

Node = collections.namedtuple("Node", 'point axis label left right')

class KDTree(object):
    #Modificaciones:
    # Datos de entrada, adaptados a los datos de prueba generados en el PC1
    # Cálculo del SqrtError, adaptado para ignorar el voto, tomando en cuenta sólo los indicadores
    # Forma general, procesamiento correcto de los datos de entrada en el nuevo formato
    # Procesamiento de listas con vecinos cercanos
    # Valor de retorno


    def __init__(self, k, objects=[]):

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

    def knn(self, destination, k):
        bestOccurrences = []
        bestSDs = []
        best = [None, None, float('inf')]
        # state of search: best point found, its label,
        # lowest squared distance

        def recursive_search(here):

            if here is None:
                return
            point, axis, label, left, right = here

            here_sd = square_distance(point, destination)

            best[:] = point, label, here_sd
            bestOccurrences.append(best[1])
            bestSDs.append(here_sd)

            if len(bestSDs) > k:
                idx = bestSDs.index(max(bestSDs))
                bestOccurrences.pop(idx)
                bestSDs.pop(idx)


            diff = destination[axis] - point[axis]
            close, away = (left, right) if diff <= 0 else (right, left)

            recursive_search(close)
            if diff ** 2 < min(bestSDs):
                recursive_search(away)

        recursive_search(self.root)
        return bestOccurrences, bestSDs



def main(maxLeafSize = 4, dataQuant = 10, dataIndicators = 34, dataSet = [], destinationSet = [], k = 2):
    
    # 

    dataSet = dataSet if dataSet else genDataSet(dataQuant,dataIndicators)
    destinationSet = destinationSet if destinationSet else [[59.0, 288054.0, 44.61999893, 6455.715081, 1.0, 1.0, 1.0, 81903.0, 3.499981686, 0.0, 
                                            0.0, 1.0, 99.43032009, 98.47701215, 9.876764995, 10.33264708, 9.200082438, 31.12624548,
                                            16.79944234, 88.54844476, 46.24617367, 9.526969815, 0.0, 0.0, 71.21249737, 44.16824899,
                                            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 6.0, 'ACCION CIUDADANA']]



    tree = KDTree(maxLeafSize, dataSet)
    precision = 0

    for testPerson in destinationSet:
        # For each person make the search of the N nearest neighbors
        bestOccurrences, bestSDs = tree.knn(testPerson, k)
        
        #sort both lists
        bestOccurrences = [x for _,x in sorted(zip(bestSDs, bestOccurrences))]
        bestSDs = sorted(bestSDs)

        #Figure out which political party is the plurality > the prediction
        partidos = [g08.PARTIDOS[int(bestOccurrences[i])] for i in range(len(bestOccurrences))]
        pluralidad =  max(set(bestOccurrences), key = bestOccurrences.count)
        
        # MIN SQUARED ERROR minsq = min(square_distance(p, destinationSet[0]) for p in dataSet)


        #print(pluralidad, " vs ", testPerson[-1])
        if pluralidad == testPerson[-1]:
            precision += 1


    # Define total precision
    precision = precision / len(destinationSet)
    print("Total precision: ", precision)
    return tree, bestOccurrences, bestSDs, precision





dataX = getParsedData(8000)
dataY = getParsedData(2000)

main(k = 4, dataSet = dataX, destinationSet = dataY)

