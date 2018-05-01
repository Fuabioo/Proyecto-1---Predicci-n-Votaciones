import csv
import pytest
import numpy
import math
import random
import argparse
from pprint import pprint
import os
acts_file = "ACTASFINAL.csv"
indicators_file = "INDICADORES.csv"
from importlib.machinery import SourceFileLoader
from io import StringIO


testing = False
riggedRandom = 0

#Definición de rangos para asignación aleatoria de cantones
cantones = [["SAN JOSE", 1, 409],
            ["ESCAZU", 410, 490],
            ["DESAMPARADOS", 491, 779],
            ["PURISCAL", 780, 851],
            ["TARRAZU", 852, 879],
            ["ASERRI", 880, 962],
            ["MORA", 963, 1009],
            ["GOICOECHEA", 1010, 1165],
            ["SANTA ANA", 1166, 1229],
            ["ALAJUELITA", 1230, 1317],
            ["VAZQUEZ DE CORONADO", 1318, 1409],
            ["ACOSTA", 1410, 1452],
            ["TIBAS", 1453, 1548],
            ["MORAVIA", 1549, 1625],
            ["MONTES DE OCA", 1626, 1699],
            ["TURRUBARES", 1700, 1715],
            ["DOTA", 1716, 1729],
            ["CURRIDABAT", 1730, 1815],
            ["PEREZ ZELEDON", 1816, 2066],
            ["LEON CORTEZ CASTRO", 2067, 2092],
            ["ALAJUELA", 2093, 2463],
            ["SAN RAMON", 2464, 2593],
            ["GRECIA", 2594, 2713],
            ["SAN MATEO", 2714, 2725],
            ["ATENAS", 2726, 2765],
            ["NARANJO", 2766, 2828],
            ["PALMARES", 2829, 2881],
            ["POAS", 2882, 2922],
            ["OROTINA", 2923, 2953],
            ["SAN CARLOS", 2954, 3181],
            ["ZARCERO", 3182, 3203],
            ["VALVERDE VEGA", 3204, 3232],
            ["UPALA", 3233, 3301],
            ["LOS CHILES", 3302, 3332],
            ["GUATUSO", 3333, 3358],
            ["CARTAGO", 3359, 3575],
            ["PARAISO", 3576, 3661],
            ["LA UNION", 3662, 3788],
            ["JIMENEZ", 3789, 3815],
            ["TURRIALBA", 3816, 3949],
            ["ALVARADO", 3950, 3971],
            ["OREAMUNO", 3972, 4035],
            ["EL GUARCO", 4036, 4098],
            ["HEREDIA", 4099, 4265],
            ["BARVA", 4266, 4325],
            ["SANTO DOMINGO", 4326, 4386],
            ["SANTA BARBARA", 4387, 4434],
            ["SAN RAFAEL", 4435, 4498],
            ["SAN ISIDRO", 4499, 4531],
            ["BELEN", 4532, 4564],
            ["FLORES", 4565, 4593],
            ["SAN PABLO", 4594, 4632],
            ["SARAPIQUI", 4633, 4711],
            ["LIBERIA", 4712, 4798],
            ["NICOYA", 4799, 4904],
            ["SANTA CRUZ", 4905, 4996],
            ["BAGACES", 4997, 5028],
            ["CARRILLO", 5029, 5076],
            ["CANHAS", 5077, 5118],
            ["ABANGARES", 5119, 5157],
            ["TILARAN", 5158, 5195],
            ["NANDAYURE", 5196, 5225],
            ["LA CRUZ", 5226, 5257],
            ["HOJANCHA", 5258, 5276],
            ["PUNTARENAS", 5277, 5470],
            ["ESPARZA", 5471, 5518],
            ["BUENOS AIRES", 5519, 5597],
            ["MONTES DE ORO", 5598, 5623],
            ["OSA", 5624, 5682],
            ["QUEPOS", 5683, 5727],
            ["GOLFITO", 5728, 5795],
            ["COTO BRUS", 5796, 5866],
            ["PARRITA", 5867, 5900],
            ["CORREDORES", 5901, 5970],
            ["GARABITO", 5971, 5992],
            ["LIMON", 5993, 6130],
            ["POCOCI", 6131, 6310],
            ["SIQUIRRES", 6311, 6396],
            ["TALAMANCA", 6397, 6442],
            ["MATINA", 6443, 6486],
            ["GUACIMO", 6487, 6542]]

#Definición votos posibles (Incluyendo votos válidos, nulos y blancos)
partidos = ["ACCESIBILIDAD SIN EXCLUSION",
            "ACCION CIUDADANA",
            "ALIANZA DEMOCRATA CRISTIANA",
            "DE LOS TRABAJADORES",
            "FRENTE AMPLIO",
            "INTEGRACION NACIONAL",
            "LIBERACION NACIONAL",
            "MOVIMIENTO LIBERTARIO",
            "NUEVA GENERACION",
            "RENOVACION COSTARRICENSE",
            "REPUBLICANO SOCIAL CRISTIANO",
            "RESTAURACION NACIONAL",
            "UNIDAD SOCIAL CRISTIANA",
            "VOTOS NULOS",
            "VOTOS BLANCOS"]

#Definición de provincias
provincias = ["SAN JOSE",
              "ALAJUELA",
              "CARTAGO",
              "HEREDIA",
              "GUANACASTE",
              "PUNTARENAS",
              "LIMON"]

votes = [1404242,
         848146,
         490903,
         433677,
         326953,
         410929,
         386862]

def parse_args():
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--indicadores', nargs="+", default=['default'], help='Archivo csv. Ej: indicadores.csv')
    parser.add_argument('--actas', nargs="+", default=['default'], help='Archivo csv. Ej: actas.csv')
    return parser.parse_args()

def load_data():
    args = parse_args()
    global indicadores
    global actas_final
    path = os.path.abspath(__file__)
    path = path.replace('g08.py','g08_indicadores.py')
    indicadores = SourceFileLoader("g08_indicadores.py", path).load_module()
    path = os.path.abspath(__file__)
    path = path.replace('g08.py','g08_actas.py')
    actas_final = SourceFileLoader("g08_actas.py", path).load_module()
    if args.indicadores[0] != 'default' and args.actas[0] != 'default':
        indicadores.INDICADORES = open(args.indicadores[0], 'r')
        actas_final.ACTAS_FINAL = open(args.actas[0], 'r')
    elif args.indicadores[0] == 'default' and args.actas[0] != 'default':
        actas_final.ACTAS_FINAL = open(args.actas[0], 'r')
    elif args.actas[0] == 'default' and args.indicadores[0] != 'default':
        indicadores.INDICADORES = open(args.indicadores[0], 'r')

def csv2mat():
    matrix = [[]]
    for i in range(1, 13):
        with open("ActaSesion" + str(i) + ".csv", 'r') as f:
            reader = csv.reader(f)
            act = list(reader)
        if i == 1:
            matrix = act
        else:
            act = numpy.delete(act, (0), axis=1)
            matrix = numpy.hstack((matrix, act))

    matrix = numpy.delete(matrix, slice(0, 2), axis=0)
    i = 1
    while (i < len(matrix[0])):
        for j in range(len(cantones)):
            if int(cantones[j][1]) <= int(matrix[0][i]) <= int(cantones[j][2]):
                matrix[0][i] = cantones[j][0]
                break
        if matrix[0][i].isdigit():
            matrix = numpy.delete(matrix, (i), axis=1)
            i -= 1
        i += 1

    fixed_matrix = matrix[:, [0]]
    for i in range(1, len(matrix[0])):
        find = False
        for j in range(len(fixed_matrix[0])):
            if fixed_matrix[0][j] == matrix[0][i]:
                find = True
                for k in range(1, len(matrix)):
                    fixed_matrix[k][j] = str(
                        int(fixed_matrix[k][j]) + int(matrix[k][i]))
        if find == False:
            fixed_matrix = numpy.hstack((fixed_matrix, matrix[:, [i]]))

    numpy.savetxt("ACTASFINAL.csv", fixed_matrix, delimiter=",", fmt="%s")
    print("Procesamiento terminado")


def round_retain_total(myOriginalList):
    originalTotal = round(sum(myOriginalList), 0)
    myRoundedList = numpy.array(myOriginalList).round(0)
    newTotal = myRoundedList.sum()
    error = originalTotal - sum(myRoundedList)
    n = int(round(error))
    myNewList = myRoundedList
    for _, i in sorted(((myOriginalList[i] - myRoundedList[i], i)
                        for i in range(len(myOriginalList))), reverse=n > 0)[:abs(n)]:
        myNewList[i] += math.copysign(1, n)
    myNewList = list(map(int, myNewList))
    return myNewList


'''
    #Dosent work at all, sometimes generate 101 instead of 100
    N = sum(xs)
    Rs = [round(x) for x in xs]
    K = N - sum(Rs)
    fs = [x - round(x) for x in xs]
    indices = [i for order, (e, i) in enumerate(reversed(sorted((e,i) for i,e in enumerate(fs)))) if order < K]
    ys = [R + 1 if i in indices else R for i,R in enumerate(Rs)]
    return ys
'''


def get_att(percent=50):
    if testing:
        percent = riggedRandom
    random.seed()
    return int(random.randrange(100) < float(percent))


def get_vote(arr):
    val = int(arr[17])
    choose = random.randrange(int(arr[17]))
    arr = numpy.concatenate((arr[1:14], arr[15:17]))

    for i in range(len(arr)):
        choose -= int(arr[i])
        if choose <= 0:
            return partidos[i]


def generar_muestra_provincia(n, nombre_provincia):
    #print("Muestra: " + nombre_provincia)
    reader = csv.reader(StringIO(actas_final.ACTAS_FINAL))
    acts = list(reader)
    reader = csv.reader(StringIO(indicadores.INDICADORES))
    indicators = list(reader)

    index = provincias.index(nombre_provincia) * 32
    total = 0
    totals = []
    for i in range(1, len(indicators[index + 1])):

        total += float(indicators[index + 1][i])
        totals += [float(indicators[index + 1][i])]

    for i in range(len(totals)):
        totals[i] = totals[i] / total * n

    totals = round_retain_total(totals)
    population = []
    for i in range(1, len(indicators[index])):
        for j in range(1, len(acts[0])):
            if indicators[index][i] == acts[0][j]:
                for k in range(totals[i - 1]):
                    A = numpy.array(acts)
                    hombres_ratio = float(indicators[index + 5][i])
                    hombres = (hombres_ratio * 100) / (hombres_ratio + 100)

                    population += [[#Demo-Geográficas
                                    (indicators[index][i]),                         # Canton
                                    float(indicators[index+1][i]),                  # Población total
                                    float(indicators[index+2][i]),                  # Superficie
                                    float(indicators[index+3][i].replace(" ","")),  # Densidad Poblacional (Estatico)
                                    get_att(indicators[index + 4][i]),              # Personas que viven en zona urbana
                                    get_att(hombres),                               # Hombre/Mujeres
                                    get_att(indicators[index + 6][i]),              # Relacion de dependencia
                                    float(indicators[index+7][i]),                  # Viviendas individuales (Estatico)
                                    float(indicators[index+8][i]),                  # Promedio de ocupantes (Estatico)
                                    get_att(indicators[index + 9][i]),              # Porcentaje de viviendas en buen estado
                                    get_att(indicators[index + 10][i]),             # Porcentaje de viviendas hacinadas
                                    #Educativas
                                    get_att(indicators[index + 11][i]),             # Porcentaje de alfabetismo
                                    float(indicators[index + 12][i]),
                                    float(indicators[index + 13][i]),
                                    float(indicators[index + 14][i]),               # Escolaridad promedio
                                    float(indicators[index + 15][i]),               # 25 a 49 años
                                    float(indicators[index + 16][i]),               # 50+ años
                                    float(indicators[index + 17][i]),               # Porcentaje de asistencia a la educaci¢n regular
                                    float(indicators[index + 18][i]),               # Menor de 5 anhos
                                    float(indicators[index + 19][i]),               # 5 a 17 anhos
                                    float(indicators[index + 20][i]),               # 18 a 24 anhos
                                    float(indicators[index + 21][i]),               # 25 y m s anhos
                                    #Económicas
                                    get_att(indicators[index + 22][i]),             # Fuera de la fuerza de trabajo
                                    get_att(indicators[index + 23][i]),             # Tasa neta de participacion
                                    float(indicators[index + 24][i]),               # Hombres
                                    float(indicators[index + 25][i]),               # Mujeres
                                    get_att(indicators[index + 26][i]),             # Porcentaje de poblacion ocupada no asegurada
                                    #Sociales
                                    get_att(indicators[index + 27][i]),             # Porcentaje de poblacion nacida en el extranjero
                                    get_att(indicators[index + 28][i]),             # Porcentaje de poblacion con discapacidad
                                    get_att(indicators[index + 29][i]),             # Porcentaje de poblacion no asegurada
                                    get_att(indicators[index + 30][i]),             # Porcentaje de hogares con jefatura femenina
                                    get_att(indicators[index + 31][i]),             # Porcentaje de hogares con jefatura compartida
                                    nombre_provincia,                               # Provincia
                                    get_vote(A[:, j])]]                             # Voto
                    

    return population


def generar_muestra_pais(n):
    total = 0
    totals = []
    for i in range(len(votes)):
        total += float(votes[i])
        totals += [float(votes[i])]

    for i in range(len(totals)):
        totals[i] = totals[i] / total * n
    totals = round_retain_total(totals)
    population = []
    for i in range(len(totals)):
        population += generar_muestra_provincia(totals[i], provincias[i])

    return population


def show_percentages(population):
    print("Porcentajes:\n")
    percents = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(population)):
        for j in range(len(partidos)):
            if population[i][len(population[0])-1] == partidos[j]:
                percents[j] += 1

    for j in range(len(partidos)):
        print(partidos[j] + ":  " + str((percents[j] / len(population) * 100)))

    return population


def show_percentages_indicator(population, indicator):
    position = get_position(indicator)
    total = 0.0
    for i in range(len(population)):
        total += population[i][position]
        
    print(indicator+": "+str((total/len(population))))
    return (total/len(population))

def show_percentages_indicator_partido(population, indicator, partido):
    position = get_position(indicator)
    total_indicador = 0.0
    total_partido = 0
    for i in range(len(population)):
        if (population[i][1]==partido):
            total_partido+=1
            total_indicador += population[i][position]


    print(partido + " - " +indicator+": "+str((total_indicador/total_partido))+"%")
    return total_indicador/total_partido


def get_position(indicator):
    return {"URBANIDAD": 3,
            "HOMBRES": 5,
            "ALFABETIZACION": 11,
            "ESCOLARIDAD": 14,
            "ASISTENCIA": 17,
            "PARTICIPACION": 23
            
        }[indicator]



load_data()

if __name__ == '__main__':

    
    
    load_data()
    
    #csv2mat()
    #show_percentages(generar_muestra_provincia(100,"CARTAGO"))
    #show_percentages(generar_muestra_pais(200000))

    
    
    #MUESTRA
    muestra1 = generar_muestra_pais(10)

    #PORCENTAJES
    #show_percentages(muestra1)
    
    show_percentages_indicator(muestra1, "URBANIDAD")
    show_percentages_indicator(muestra1, "HOMBRES")
    show_percentages_indicator(muestra1, "ALFABETIZACION")
    show_percentages_indicator(muestra1, "ESCOLARIDAD")
    show_percentages_indicator(muestra1, "ASISTENCIA")
    show_percentages_indicator(muestra1, "PARTICIPACION")

    print(muestra1[0])
    #show_percentages_indicator_partido(muestra1, "PARTICIPACION", "RESTAURACION NACIONAL")
