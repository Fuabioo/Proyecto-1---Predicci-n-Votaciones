import collections
import itertools
import math
import random
import numpy
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tec.ic.ia.p1 import g08_data
from tec.ic.ia.pc1 import g08

def getParsedData(n):

    data1round, data2round, data2round1 = g08_data.shaped_data_no_bin2(n)

    return data1round, data2round, data2round1

"""
def genDataSet(n, m):
    dataSet = []
    dataSet = [list(random.randint(0,1) for x in range(m)) for y in range(n)]
    for i in dataSet:
        if i[3]%2 == 0:
            i[-1] = 'PAC'
        else:
            i[-1] = 'RN'


    return dataSet
"""

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


def calculateTreeData(dataSet, testSet, maxLeafSize, k):
    
    
    predictionList = []

    tree = 0
    print("Generating tree")
    tree = KDTree(maxLeafSize, dataSet)
    

    print("Processing")

    precision = 0
    for testPerson in testSet:
        # For each person make the search of the N nearest neighbors
        bestOccurrences, bestSDs = tree.knn(testPerson, k)
        predict =  max(set(bestOccurrences), key = bestOccurrences.count)
        predictionList.append(predict)
        if predict == testPerson[-1]:
            precision += 1

    precision = precision / len(testSet)
    print("Total precision: ", precision)
    return tree, precision, predictionList

def kdknn(allSets, maxLeafSize = 10, k = 5, testPercent = 20):
    
    # 
    #

    if testPercent>1:
        testPercent /= 100

    if not allSets:
        return

    set1 = allSets[0][0]
    tSet1 = allSets[0][1]
    set2 = allSets[1][0]
    tSet2 = allSets[1][1]
    set3 = allSets[2][0]
    tSet3 = allSets[2][1]

    destinationSet = tSet3

    #Ronda 1
    print("Creating tree round 1")
    tree1, precision1, predictions1 = calculateTreeData(set1, tSet1, maxLeafSize, k)

    #Ronda 2 sin ronda 1
    print("Creating tree round 2 without round 1")
    tree2, precision2, predictions2 = calculateTreeData(set2, tSet2, maxLeafSize, k)

    #Ronda 2 con ronda 1
    print("Creating tree round 2 with round 1")
    tree3, precision3, predictions3 = calculateTreeData(set3, tSet3, maxLeafSize, k)


    return destinationSet, [tree1, tree2, tree3] , [predictions1, predictions2, predictions3], [precision1, precision2, precision3]


def processSplittedData(splitted, index):
    # Lists with the datasets splitted
    datasetPerRound = []

    trainWith = splitted.copy()
    testWith = trainWith.pop(index)
    trainWith = list(itertools.chain.from_iterable(trainWith))

    datasetPerRound.append(trainWith)
    datasetPerRound.append(testWith)

    return datasetPerRound

def switchColumns(dataSet, x, y):
    for i in range(len(dataSet)):
        dataSet[i][x], dataSet[i][y] = dataSet[i][y], dataSet[i][x] 
    return dataSet

def crossValidate(parts = 4, datasetSize = 10026, finalPercent = 20):

    # Get full datasets 1, 2, 3 (The same dataset expressed in different ways)
    data1, data2, data3 = getParsedData(datasetSize)
    print(len(data1))
    trees1 = []
    trees2 = []
    trees3 = []
    predictions1 = []
    predictions2 = []
    predictions3 = []
    precisions1 = []
    precisions2 = []
    precisions3 = []

    # 0, 1, 2, 7 || 23, 29
    #data1 = switchColumns(data1.copy(), 0, 1 )
    data1 = switchColumns(data1.copy(), 2, 23 )
    data1 = switchColumns(data1.copy(), 7, 29 )

    #data2 = switchColumns(data2.copy(), 0, 1 )
    data2 = switchColumns(data2.copy(), 2, 23 )
    data2 = switchColumns(data2.copy(), 7, 29 )

    #data3 = switchColumns(data3.copy(), 0, 1 )
    data3 = switchColumns(data3.copy(), 2, 23 )
    data3 = switchColumns(data3.copy(), 7, 29 )
    data3 = switchColumns(data3.copy(), 0, len(data3[0])-2)

    # Format data for cross validation
    parts = int(len(data1)//parts)


    # Cross validate data 1
    data1split = [data1[i:i+parts] for i  in range(0, len(data1), parts)]
    data2split = [data2[i:i+parts] for i  in range(0, len(data2), parts)]
    data3split = [data3[i:i+parts] for i  in range(0, len(data3), parts)]
    #list(itertools.chain.from_iterable(lists))
    
    for i in range(len(data3split)):
        
        allDatasets = []
        

        print("TESTING WITH: ", i)

        

        # Round 1
        datasetPerRound = processSplittedData(data1split, i)
        allDatasets.append(datasetPerRound)

        # Round 2 without
        datasetPerRound = processSplittedData(data2split, i)
        allDatasets.append(datasetPerRound)

        # Round 2 with
        datasetPerRound = processSplittedData(data3split, i)
        allDatasets.append(datasetPerRound)
        
        _, trees , _, precisions = kdknn(allSets = allDatasets)

        trees1.append(trees[0])
        trees2.append(trees[1])
        trees3.append(trees[2])
        precisions1.append(precisions[0])
        precisions2.append(precisions[1])
        precisions3.append(precisions[2])


    finalSet1, finalSet2, finalSet3 = getParsedData(datasetSize)
    bestTree1, bestTree2, bestTree3 = getBestTrees(trees1, trees2, trees3, precisions1, precisions2, precisions3)
    finalTests([bestTree1, bestTree2, bestTree3], [finalSet1, finalSet2, finalSet3])
    return 


def finalTests(bestTrees, dataSets, k=5):
    dataSetIndex = 0
    for dataSet in dataSets:

        
        tree = bestTrees[dataSetIndex]
        print("Processing tests")
        precision = 0
        for testPerson in dataSet:
            bestOccurrences, bestSDs = tree.knn(testPerson, k)
            predict =  max(set(bestOccurrences), key = bestOccurrences.count)
            if predict == testPerson[-1]:
                precision += 1
                


        precision = precision / len(dataSet)
        dataSetIndex += 1
        print("Total precision: ", precision)
    


def getBestTrees(trees1, trees2, trees3, precisions1, precisions2, precisions3):
    print("Best precision 1", max(precisions1))
    ind1 = precisions1.index(max(precisions1))
    print("Best precision 2", max(precisions2))
    ind2 = precisions2.index(max(precisions2))
    print("Best precision 3", max(precisions3))
    ind3 = precisions3.index(max(precisions3))


    return trees1[ind1], trees2[ind2], trees3[ind3]

crossValidate(parts = 5, datasetSize = 1003)
