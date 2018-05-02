from tec.ic.ia.p1 import g08_kdtrees 

# Funciones de testing unitario para el Proyecto 1 de Inteligencia Artificial
# 



# VARIAS
def test_argparse():
	return





# REDES NEURONALES
def test_neural():
	return






# REGRESIÓN LINEAL
def test_linear():
	return






# ARBOL DE DECISIÓN
def test_decision():
	return





# K-D KNN
def test_knn():
	# Votos para PAC, PLN y RN
	dataset = [[0.0,0.0,0.0,0.0, 1.0], [1.0,1.0,1.0,1.0, 6.0], [2.0,2.0,2.0,2.0, 12.0]]
	testSubject = [[2.0,2.0,1.0,2.0, 12.0]] 
	k = 1
	leafSize = 2

	# El vecino más cercano es [2,2,2,2, 'RESTAURACION NACIONAL']
	# Se espera una predicción de 'RESTAURACION NACIONAL' (12) y una precisión de 1

	_, occurrences ,_, precision = g08_kdtrees.main(dataSet = dataset, destinationSet = testSubject, k=k, maxLeafSize = leafSize)
	plurality =  max(set(occurrences), key = occurrences.count)

	assert(precision == 1)
	assert(plurality == 12.0) #restauración nacional

# SVM



test_knn()