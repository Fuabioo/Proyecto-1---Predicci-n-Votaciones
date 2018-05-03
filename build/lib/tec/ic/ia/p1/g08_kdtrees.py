import collections
import itertools
import math
import random
import numpy
from tec.ic.ia.p1 import g08_data
from tec.ic.ia.pc1 import g08

def getParsedData(n):

    data = g08_data.shaped_data_no_bin2(n,1).tolist()
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



def main(dataQuant = 1000, dataSet = [], destinationSet = [], maxLeafSize = 10, k = 2):
    
    # 
    #
    print("Generating dataset if not provided")
    dataSet = dataSet if dataSet else getParsedData(dataQuant - (dataQuant/5))
    destinationSet = destinationSet if destinationSet else getParsedData(dataQuant/5)


    print("Creating tree")
    tree = KDTree(maxLeafSize, dataSet)
    precision = 0

    print("Processing")
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
    print(bestOccurrences)
    return tree, bestOccurrences, bestSDs, precision

"""
def separate(lst, parts):
    ret = []
    quant = len(lst)/parts
    for i in range(parts):
        base = int(quant*i)
        ending = int(quant*i+quant)
        ret.append(lst[base : ending])
    
    return ret

def crossValidate(data, parts = 10):

<<<<<<< HEAD
    error = 0

    # Separate into PARTS 
    dataParts = separate(data, parts)
    for i in range(parts):


        #Define dataSet training set
        dataSet = dataParts.copy()

        #Define dataTest testing set
        dataTest = dataSet.pop(i)
=======
>>>>>>> 33e94ed734c9c07b36ff9692e467e915bac873b0

        #Format dataSet before processing
        dataSet = list(itertools.chain.from_iterable(dataSet))

        main()

    return error
"""
dataX = getParsedData(8000)

dataY = getParsedData(2000)

<<<<<<< HEAD
main(k = 40, dataSet = dataX, destinationSet = dataY)


"""
dataX = getParsedData(10)
crossValidate(dataX, 2)
"""
=======
main(k = 4, dataSet = dataX, destinationSet = dataY)
>>>>>>> 33e94ed734c9c07b36ff9692e467e915bac873b0
