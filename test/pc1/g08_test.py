import pytest
import tec.ic.ia.pc1.g08 as g08
g08.load_data()




#----------------------------------------------DEFINICIONES----------------------------------------------
#Tamaño de la muestra
@pytest.fixture
def N():
    return 1000 #Modificar a conveniencia

#Indicadores
@pytest.fixture
def indicators():
    return ["HOMBRES",
            "ALFABETIZACION",
            "PARTICIPACION"
            ] #Modificar a conveniencia


#Tolerancia
@pytest.fixture
def r():
    return 0.05 #Modificar a conveniencia

@pytest.fixture
def provinces():
    return ["SAN JOSE",
            "ALAJUELA",
            "CARTAGO",
            "HEREDIA",
            "GUANACASTE",
            "PUNTARENAS",
            "LIMON"]#Modificar a conveniencia
#----------------------------------------------DEFINICIONES----------------------------------------------




#----------------------------------------------PAIS------------------------------------------------------
#Prueba de distribución de PROVINCIAS
            
#Verifica la correcta distribucion de personas por provincia
def test_distribucion_por_provincia(N):
    dicc= {"SAN JOSE": 0,
           "ALAJUELA": 1,
           "CARTAGO": 2,
           "HEREDIA": 3,
           "GUANACASTE": 4,
           "PUNTARENAS": 5,
           "LIMON": 6}

    muestraPop = [0,0,0,0,0,0,0]
    population = [1404242, #SJO
                  848146,  #ALA
                  490903,  #CAR
                  433677,  #HER
                  326953,  #GUA
                  410929,  #PUN
                  386862]  #LIM

    totalPop = sum(population)

    distribucionPProvincia = [0,0,0,0,0,0,0]
    distribucionPProvinciaMuestra = [0,0,0,0,0,0,0]

    for i in range(len(population)):
        distribucionPProvincia[i] = population[i]/totalPop

    muestra1 = g08.generar_muestra_pais(N)

    for i in range(len(muestra1)):
        muestraPop[dicc[muestra1[i][32]]] +=1

    for i in range(len(muestraPop)):
        distribucionPProvinciaMuestra[i] = muestraPop[i]/N

    for i in range(len(population)):
        assert (distribucionPProvincia[i]-0.01 < distribucionPProvinciaMuestra[i] <distribucionPProvincia[i]+0.01)




#Pruebas de aleatoriedad PAIS

#Forma: test_pais_X
#   donde X es el porcentaje modificado para indicadores directamente dependientes de porcentajes
#   en el archivo de indicadores


def test_pais_0(N, indicators):
    
    g08.testing = True
    g08.riggedRandom = 0
    
    muestra = g08.generar_muestra_pais(N)

    for i in range(len(indicators)):
        assert(g08.show_percentages_indicator(muestra, indicators[i])==0)

def test_pais_10(N, indicators, r):

    
    g08.testing = True
    g08.riggedRandom = 10
    
    muestra = g08.generar_muestra_pais(N)

    for i in range(len(indicators)):
        assert(0.10-r <g08.show_percentages_indicator(muestra, indicators[i])< 0.10+r)
    
def test_pais_50(N, indicators, r):

    g08.testing = True
    g08.riggedRandom = 50
    
    muestra = g08.generar_muestra_pais(N)
    for i in range(len(indicators)):
        assert(0.5-r <g08.show_percentages_indicator(muestra, indicators[i])< 0.5+r)
    
def test_pais_90(N, indicators, r):

    g08.testing = True
    g08.riggedRandom = 90
    
    muestra = g08.generar_muestra_pais(N)
    for i in range(len(indicators)):
        assert(0.9-r < g08.show_percentages_indicator(muestra, indicators[i]) < 0.9+r)

def test_pais_100(N, indicators):
    
    g08.testing = True
    g08.riggedRandom = 100
    
    muestra = g08.generar_muestra_pais(N)
    for i in range(len(indicators)):
        assert(g08.show_percentages_indicator(muestra, indicators[i])==1.0)
#----------------------------------------------PAIS------------------------------------------------------



#----------------------------------------------PROVINCIA-------------------------------------------------
#Pruebas de aleatoriedad PROVINCIA

#Forma: test_provincia_X
#   donde X es el porcentaje modificado para indicadores directamente dependientes de porcentajes
#   en el archivo de indicadores

def test_provincia_0(N, indicators):
    
    g08.testing = True
    g08.riggedRandom = 0
    
    muestra = g08.generar_muestra_provincia(N, "SAN JOSE")

    for i in range(len(indicators)):
        assert(g08.show_percentages_indicator(muestra, indicators[i])==0)

def test_provincia_10(N, indicators, r):

    
    g08.testing = True
    g08.riggedRandom = 10
    
    muestra = g08.generar_muestra_provincia(N, "CARTAGO")

    for i in range(len(indicators)):
        assert(0.10-r <g08.show_percentages_indicator(muestra, indicators[i])< 0.10+r)
    
def test_provincia_50(N, indicators, r):

    g08.testing = True
    g08.riggedRandom = 50
    
    muestra = g08.generar_muestra_provincia(N, "ALAJUELA")
    for i in range(len(indicators)):
        assert(0.5-r <g08.show_percentages_indicator(muestra, indicators[i])< 0.5+r)
    
def test_provincia_90(N, indicators, r):

    g08.testing = True
    g08.riggedRandom = 90
    
    muestra = g08.generar_muestra_provincia(N, "HEREDIA")
    for i in range(len(indicators)):
        assert(0.9-r < g08.show_percentages_indicator(muestra, indicators[i]) < 0.9+r)

def test_provincia_100(N, indicators):
    
    g08.testing = True
    g08.riggedRandom = 100
    
    muestra = g08.generar_muestra_provincia(N, "CARTAGO")
    for i in range(len(indicators)):
        assert(g08.show_percentages_indicator(muestra, indicators[i])==1.0)

#Prueba de pertenencia a PROVINCIA
        
#Revisa que no haya personas de provincias diferentes al generar una provincia
#específica
def test_pertenencia(N, provinces):

    for province in provinces:
        muestra = g08.generar_muestra_provincia(N//4, province)

        for persona in muestra:
            assert(persona[32]==province)



