import collections
import math
import numpy
import sys
import g08
from pptree import *
sys.setrecursionlimit(5000)

error_margin = 0.15
data = list(numpy.array(g08.generar_muestra_pais(50)))
attr = list(range(len(data[0])))
names = ["Canton","Población total","Superficie","Densidad Poblacional",
        "Personas en zona urbana","Hombre/Mujer", "Dependiente",
        "Viviendas individuales", "Promedio de ocupantes","Viviendas en buen estado",
        "viviendas hacinadas","Alfabetismo","A","B","Escolaridad promedio",
        "25 a 49 años","50+ años","Asistencia a educaci¢n regular","Menor de 5 anhos",
        "5 a 17 anhos","18 a 24 anhos","25 y mas anhos","Fuera de la fuerza de trabajo",
        "Tasa neta de participacion","Hombres","Mujeres","Porcentaje de Mujeres",
        "nacida en el extranjero","con discapacidad","No asegurado","jefatura femenina",
        "jefatura compartida","Provincia","Voto"]

class Tree(object):
    def __init__(self, data, percents,head=None):
        self.children = []
        self.data = data
        self.percents = percents
        if head:
            head.children.append(self)
            
    def add_child(self, son):
        self.children.append(son)

    def __str__(self):
        return self.function
    
def majority_value(data, target_attr):
    data = data[:]
    return most_frequent([record[target_attr] for record in data])

def most_frequent(lst):
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def unique(lst):
    lst = lst[:]
    unique_lst = []

    # Cycle through the list and add each value to the unique list only once.
    for item in lst:
        if unique_lst.count(item) <= 0:
            unique_lst.append(item)
            
    # Return the list with all redundant values removed.
    return unique_lst

def get_values(data, attr):
    data = data[:]
    return unique([record[attr] for record in data])

def choose_attribute(data, attributes, target_attr, fitness):
    data = data[:]
    best_gain = float('inf')
    best_attr = None
    best_percents = None
    for attr in attributes:
        gain,percents = fitness(data, attr, target_attr)
        if (gain <= best_gain and attr != target_attr):
            best_gain = gain
            best_attr = attr
            best_percents = percents
            
    if best_gain < error_margin:
        return -1
    else:
        return best_attr,best_percents

def get_examples(data, attr, value):
    data = data[:]
    rtn_lst = []
    
    if not data:
        return rtn_lst
    else:
        record = data.pop()
        if record[attr] == value:
            rtn_lst.append(record)
            rtn_lst.extend(get_examples(data, attr, value))
            return rtn_lst
        else:
            rtn_lst.extend(get_examples(data, attr, value))
            return rtn_lst

def get_classification(record, tree):
    # If the current node is a string, then we've reached a leaf node and
    # we can return it as our answer
    if type(tree) == type("string"):
        return tree

    # Traverse the tree further until a leaf node is found.
    else:
        attr = tree.keys()[0]
        t = tree[attr][record[attr]]
        return get_classification(record, t)

def classify(tree, data):
    data = data[:]
    classification = []
    
    for record in data:
        classification.append(get_classification(record, tree))

    return classification

def create_decision_tree(data, attributes, target_attr, fitness_func, names, head=None, val = ""):
    data = data[:]
    vals = [record[target_attr] for record in data]
    default = majority_value(data, target_attr)

    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if not data or (len(attributes) - 1) <= 0:
        return default
    # If all the records in the dataset have the same classification,
    # return that classification.
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        # Choose the next best attribute to best classify our data
        best = choose_attribute(data, attributes, target_attr,
                                fitness_func)
        if best == -1:
            return
        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        # We use the collections.defaultdict function to add a function to the
        # new tree that will be called whenever we query the tree with an
        # attribute that does not exist.  This way we return the default value
        # for the target attribute whenever, we have an attribute combination
        # that wasn't seen during training.

        tree = Tree(val +"-> "+ names[best[0]],best[1],head)


        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in get_values(data, best[0]):
            # Create a subtree for the current value under the "best" field
            print([val])
            subtree = create_decision_tree(
                get_examples(data, best[0], val),
                [attr for attr in attributes if attr != best[0]],
                target_attr,
                fitness_func,names ,tree, val)


    return tree


def entropy(data, target_attr):
    val_freq     = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in data:
        if ((record[target_attr]) in val_freq):
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]]  = 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
        
    return data_entropy,val_freq


def gain(data, attr, target_attr):
    val_freq       = {}
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    
    for record in data:
        if ((record[attr]) in val_freq):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]]  = 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob        = val_freq[val] / sum(val_freq.values())
        data_subset     = [record for record in data if record[attr] == val]
        subset_entropy += val_prob * entropy(data_subset, target_attr)[0]

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    
    res = (entropy(data, target_attr) )

    return res


tree = (create_decision_tree(data, attr, len(data[0])-1, gain, names))
print_tree(tree ,'children' , 'data' )
print_tree(tree ,'children' , 'percents' )

