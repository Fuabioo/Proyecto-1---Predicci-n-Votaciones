"""
Proyecto #1 - Predicción Votaciones
"""
from argparse import ArgumentParser

from tec.ic.ia.p1 import g08_redes_neuronales
#from tec.ic.ia.p1 import g08_kdtrees
#from tec.ic.ia.p1 import g08_desicion_tree
#from tec.ic.ia.p1 import g08_regresion
from tec.ic.ia.pc1 import g08

def get_args():
    """
    Processes the arguments, returning them
    """
    parser = ArgumentParser(description="Proyect #1 - Votes predictor")

    parser.add_argument('--poblacion', default='-1', type=int, required=True)
    parser.add_argument(
        '--porcentaje-pruebas',
        default='-1',
        type=int,
        required=True)

    parser.add_argument('--provincia', default='PAIS', type=str)

    parser.add_argument('--regresion-logistica', action='store_true')
    parser.add_argument('--l1', action='store_true')
    parser.add_argument('--l2', action='store_true')

    parser.add_argument('--red-neuronal', action='store_true')
    parser.add_argument('--numero-capas', default='1', type=int)
    parser.add_argument('--unidades-por-capa', default='100', type=int)
    parser.add_argument('--funcion-activacion', default='relu', type=str)

    parser.add_argument('--arbol', action='store_true')
    parser.add_argument('--umbral-poda', default='0.1', type=float)

    parser.add_argument('--knn', action='store_true')
    parser.add_argument('--k', default='3', type=int)

    parser.add_argument('--svm', action='store_true')
    args = parser.parse_args()
    return args


def regresion_logistica(args, dataset):
    """
    Ejecucion de la regresion logica
    """
    print("Regresion con l1", args.l1, ", l2", args.l2)


def red_neuronal(args, dataset):
    """
    Ejecucion de la red neuronal
    """
    if args.numero_capas == -1:
        print("ValueError: numero-capas")
    elif args.unidades_por_capa == -1:
        print("ValueError: unidades-por-capa")
    else:
        print(
            "Red neuronal con",
            args.numero_capas,
            "capas,",
            args.unidades_por_capa,
            "unidades por capa y",
            args.funcion_activacion,
            "como funcion de activacion")
        result = g08_redes_neuronales.execute_model(args.numero_capas,
            args.unidades_por_capa,
            args.funcion_activacion,
            dataset,
            args.porcentaje_pruebas)


def arbol(args, dataset):
    """
    Ejecucion del arbol de decision
    """
    print("Arbol con umbral de poda", args.umbral_poda)


def knn(args, dataset):
    """
    Ejecucion del knn
    """
    if args.k == -1:
        print("ValueError: k")
    else:
        print("KNN con k = ", args.k)
        g08_kdtrees.main(dataQuant = args.poblacion, k = args.k)


def svm(args, dataset):
    """
    Ejecucion de la svm
    """
    print("SVM = ", args.svm)

def gen_dataset(n, provincia, sample_type=1):
    result = None
    if provincia != "PAIS":
        result = g08.generar_muestra_provincia(n,provincia,sample_type)
    else:
        result = g08.generar_muestra_pais(n,sample_type)
    return result


def run_prediction():
    """
    Runs a prediction for each model
    """
    # Load Arguments
    args = get_args()
    # Generar dataset
    dataset = gen_dataset(args.poblacion, args.provincia)
    if args.regresion_logistica:
        regresion_logistica(args, dataset)
    elif args.red_neuronal:
        red_neuronal(args, dataset)
    elif args.arbol:
        arbol(args, dataset)
    elif args.knn:
        knn(args, dataset)
    elif args.svm:
        svm(args, dataset)


if __name__ == '__main__':
    run_prediction()
