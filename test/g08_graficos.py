import numpy as np
import matplotlib.pyplot as plt
import tec.ic.ia.pc1.g08 as g08
from tec.ic.ia.pc1.g08 import generar_muestra_pais, generar_muestra_provincia

VOTOS_RECIBIDOS = 2182764
VOTOS_NULOS_BLANCOS = 26292
REAL_PAIS = [0.35, 21.63, 0.59, 0.20, 0.78, 9.54, 18.63, 1.02, 0.76, 0.57, 4.94, 24.99, 15.99, VOTOS_RECIBIDOS/(100*VOTOS_NULOS_BLANCOS)]
REAL_SJO = [0.31,23.51,0.35,0.20,0.78,9.01,17.98,1.02,0.85,0.54,4.59,22.89,17.98, VOTOS_RECIBIDOS/(100*VOTOS_NULOS_BLANCOS)]
REAL_ALA = [0.33,21.83,0.31,0.21,0.75,8.85,18.16,0.91,0.72,0.49,5.44,26.76,15.23, VOTOS_RECIBIDOS/(100*VOTOS_NULOS_BLANCOS)]
REAL_CAR = [0.39,26.43,2.16,0.19,0.76,10.87,20.17,1.03,0.86,0.71,6.36,15.02,15.06, VOTOS_RECIBIDOS/(100*VOTOS_NULOS_BLANCOS)]
REAL_HER = [0.25,27.28,0.29,0.17,0.80,8.07,17.68,1.02,0.93,0.52,3.72,21.18,18.08, VOTOS_RECIBIDOS/(100*VOTOS_NULOS_BLANCOS)]
REAL_GUA = [0.38,15.08,0.31,0.18,0.67,11.20,23.56,0.96,0.49,0.59,5.20,25.56,15.81, VOTOS_RECIBIDOS/(100*VOTOS_NULOS_BLANCOS)]
REAL_PUN = [0.39,12.02,0.36,0.19,0.80,11.51,18.50,1.07,0.56,0.56,4.85,35.54,13.64, VOTOS_RECIBIDOS/(100*VOTOS_NULOS_BLANCOS)]
REAL_LIM = [0.65,10.56,0.66,0.30,0.94,10.40,17.56,1.29,0.43,0.80,4.59,42.58,9.24, VOTOS_RECIBIDOS/(100*VOTOS_NULOS_BLANCOS)]

REAL_ACTUAL=REAL_CAR
actual = "Cartago"

def show_percentages(population):
    keys = []
    values = []
    percents = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(population)):
        for j in range(len(g08.partidos)):
            if population[i][len(population[0])-1] == g08.partidos[j]:
                percents[j] += 1

    for j in range(len(g08.partidos)):
        sub_result = []
        keys.append(g08.partidos[j])
        values.append((percents[j] / (len(population)) * 100))

    return keys, values

def average_aux(listas):
	result = []
	length = len(listas[0])
	epsilon = 0.0
	for i in range(length):
		for lista in listas:
			epsilon += lista[i]
		result.append(epsilon/length)
		epsilon = 0.0
	return result

def averge(n):
	corridas_pais = []
	result = []
	keys = []
	for i in range(10):
		if actual!="Pais": keys, values = show_percentages(generar_muestra_provincia(n, actual.upper()))
		elif actual=="Pais": keys, values = show_percentages(generar_muestra_pais(n))
		corridas_pais.append(values)
	corridas_pais = average_aux(corridas_pais)
	return keys, corridas_pais

def _print(keys, listas):
	print(actual)
	for i in range(len(keys)-2):
		line = ''
		line += keys[i]
		line += '  '
		line += str(listas[i])
		line += '  '
		line += str(REAL_ACTUAL[i])
		print(line)
	i = len(keys)-2
	line = ''
	line += keys[i]
	line += '  '
	line += str(listas[i]+listas[i+1])
	line += '  '
	line += str(REAL_ACTUAL[i])

def main(tamano_de_muestra):	
	keys, values = averge(tamano_de_muestra)
	_print(keys, values)

	plt.suptitle('Promedio de 10 corridas ('+actual+')')
	plt.ylabel('Resultados de la votacion')
	plt.xlabel('Partido')
	plt.title('Muestra de '+str(tamano_de_muestra), fontsize=16)

	n = len(keys)
	values[n-2] = values[n-2] + values[n-1]
	values = values[:-1]
	keys = keys[:-1]

	n = len(REAL_ACTUAL)

	Y = REAL_ACTUAL
	for i in range(n):
		if Y[i]<0: Y[i] = -1 * Y[i]
	
	limit = max([max(REAL_ACTUAL),max(values)]) + 5

	keys = keys[:-1]
	Y = REAL_ACTUAL
	for i in range(n):
		if Y[i]>0: Y[i] = -1 * Y[i]
	for i in range(len(keys)):
		keys[i] = keys[i].replace(' ', '\n')
	X = range(n)
	width = 1/1.5

	plt.bar(X, values, color="blue")
	plt.bar(X, Y, color="red")
	plt.xticks(X, keys, rotation=90, size=5)
	for x,y in zip(X,values):
		plt.text(x, y+0.05, '%.2f' % y, ha='center', va= 'bottom', size=7)
	for x,y in zip(X,Y):
		plt.text(x, y-2.0, '%.2f' % abs(y), ha='center', va= 'bottom', size=7)

	plt.ylim(-limit,+limit)
	#plt.show()
	plt.savefig("graficos/"+actual+'-'+str(tamano_de_muestra)+'.png')
	plt.clf()

if __name__ == '__main__':
	REAL_ACTUAL=REAL_PAIS
	actual = "Pais"
	main(1000)
	REAL_ACTUAL=REAL_SJO
	actual = "San Jose"
	main(1000)
	REAL_ACTUAL=REAL_CAR
	actual = "Cartago"
	main(1000)
	REAL_ACTUAL=REAL_ALA
	actual = "Alajuela"
	main(1000)
	REAL_ACTUAL=REAL_HER
	actual = "Heredia"
	main(1000)
	REAL_ACTUAL=REAL_PUN
	actual = "Puntarenas"
	main(1000)
	REAL_ACTUAL=REAL_GUA
	actual = "Guanacaste"
	main(1000)
	REAL_ACTUAL=REAL_LIM
	actual = "Limon"
	main(1000)
	REAL_ACTUAL=REAL_PAIS
	actual = "Pais"
	main(10000)
	REAL_ACTUAL=REAL_SJO
	actual = "San Jose"
	main(10000)
	REAL_ACTUAL=REAL_CAR
	actual = "Cartago"
	main(10000)
	REAL_ACTUAL=REAL_ALA
	actual = "Alajuela"
	main(10000)
	REAL_ACTUAL=REAL_HER
	actual = "Heredia"
	main(10000)
	REAL_ACTUAL=REAL_PUN
	actual = "Puntarenas"
	main(10000)
	REAL_ACTUAL=REAL_GUA
	actual = "Guanacaste"
	main(10000)
	REAL_ACTUAL=REAL_LIM
	actual = "Limon"
	main(10000)
	REAL_ACTUAL=REAL_PAIS
	actual = "Pais"
	main(100000)
	REAL_ACTUAL=REAL_SJO
	actual = "San Jose"
	main(100000)
	REAL_ACTUAL=REAL_CAR
	actual = "Cartago"
	main(100000)
	REAL_ACTUAL=REAL_ALA
	actual = "Alajuela"
	main(100000)
	REAL_ACTUAL=REAL_HER
	actual = "Heredia"
	main(100000)
	REAL_ACTUAL=REAL_PUN
	actual = "Puntarenas"
	main(100000)
	REAL_ACTUAL=REAL_GUA
	actual = "Guanacaste"
	main(100000)
	REAL_ACTUAL=REAL_LIM
	actual = "Limon"
	main(100000)
