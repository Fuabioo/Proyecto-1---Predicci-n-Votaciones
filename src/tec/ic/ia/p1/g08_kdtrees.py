import collections
import itertools
import math
import random
import numpy
from tec.ic.ia.p1 import g08_data
from tec.ic.ia.pc1 import g08

def getParsedData(n):

    data1round, data2round, data2round1 = g08_data.shaped_data_no_bin2(n)

    return data1round, data2round, data2round1


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



def main(allSets, maxLeafSize = 10, k = 2, testPercent = 20):
    
    # 
    #

    if testPercent>1:
        testPercent /= 100

    if not allSets:
        return

    set1 = allSets[0]
    set2 = allSets[1]
    set3 = allSets[2]
    destinationSet = set3[int(len(set3) *  (1-testPercent)) : ]

    #Ronda 1
    print("Creating tree round 1")
    tree1, precision1, predictions1 = calculateTreeData(set1, maxLeafSize, k, testPercent)

    #Ronda 2 sin ronda 1
    print("Creating tree round 2 without round 1")
    tree2, precision2, predictions2 = calculateTreeData(set2, maxLeafSize, k, testPercent)

    #Ronda 2 con ronda 1
    print("Creating tree round 2 with round 1")
    tree3, precision3, predictions3 = calculateTreeData(set3, maxLeafSize, k, testPercent)


    return destinationSet, predictions1, predictions2, predictions3


def calculateTreeData(pSet, maxLeafSize, k, testPercent):
    
    precision = 0
    dataSet = pSet[0: int(len(pSet) *  (1-testPercent))]
    testSet = pSet[int(len(pSet) *  (1-testPercent)) : ]
    tree = KDTree(maxLeafSize, dataSet)
    predictionList = []

    print("Processing")
    print("Test set length", len(testSet))
    for testPerson in testSet:

        # For each person make the search of the N nearest neighbors
        bestOccurrences, bestSDs = tree.knn(testPerson, k)
        print("Occurrences quantity: ",len(bestOccurrences))
        #sort both lists
        #bestOccurrences = [x for _,x in sorted(zip(bestSDs, bestOccurrences))]
        #bestSDs = sorted(bestSDs)

        #Figure out which political party is the plurality > the prediction
        partidos = [g08.PARTIDOS[int(bestOccurrences[i])] for i in range(len(bestOccurrences))]
        predict =  max(set(bestOccurrences), key = bestOccurrences.count)
        predictionList.append(predict)
        
        if predict == testPerson[-1]:
            precision += 1

    precision = precision / len(testSet)
    print("Total precision: ", precision)
    return tree, precision, predictionList


"""
def separate(lst, parts):
    ret = []
    quant = len(lst)/parts
    
        base = int(quant*i)
        ending = int(quant*i+quant)
        ret.append(lst[base : ending])
    
    return ret"""

def crossValidate(data, parts = 10):

    

    # Separate into PARTS 
    dataParts = separate(data, parts)
    for i in range(parts):


        #Define dataSet training set
        dataSet = dataParts.copy()

        #Define dataTest testing set
        dataTest = dataSet.pop(i)

        #Format dataSet before processing
        dataSet = list(itertools.chain.from_iterable(dataSet))

        main()

    return error




data1, data2, data3 = getParsedData(1000)
main(k = 5, allSets = (data1, data2, data3))
