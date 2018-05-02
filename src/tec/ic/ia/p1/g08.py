"""
Proyecto #1 - Predicción Votaciones
"""
from argparse import ArgumentParser


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

    parser.add_argument('--regresion-logistica', action='store_true')
    parser.add_argument('--l1', action='store_true')
    parser.add_argument('--l2', action='store_true')

    parser.add_argument('--red-neuronal', action='store_true')
    parser.add_argument('--numero-capas', default='-1', type=int)
    parser.add_argument('--unidades-por-capa', default='-1', type=int)
    parser.add_argument('--funcion-activacion', default='relu', type=str)

    parser.add_argument('--arbol', action='store_true')
    parser.add_argument('--umbral-poda', action='store_true')

    parser.add_argument('--knn', action='store_true')
    parser.add_argument('--k', default='-1', type=int)

    parser.add_argument('--svm', action='store_true')
    args = parser.parse_args()
    return args


def regresion_logistica(args):
    """
    Ejecucion de la regresion logica
    """
    print("Regresion con l1", args.l1, ", l2", args.l2)


def red_neuronal(args):
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


def arbol(args):
    """
    Ejecucion del arbol de decision
    """
    print("Arbol con umbral de poda", args.umbral_poda)


def knn(args):
    """
    Ejecucion del knn
    """
    if args.k == -1:
        print("ValueError: k")
    else:
        print("KNN con k = ", args.k)


def svm(args):
    """
    Ejecucion de la svm
    """
    print("SVM = ", args.svm)


def main():
    """
    Main execution of the program
    """
    # Load Arguments
    args = get_args()
    # for arg in vars(args):
    #    print(arg, getattr(args, arg))
    if args.regresion_logistica:
        regresion_logistica(args)
    elif args.red_neuronal:
        red_neuronal(args)
    elif args.arbol:
        arbol(args)
    elif args.knn:
        knn(args)
    elif args.svm:
        svm(args)


if __name__ == '__main__':
    main()
