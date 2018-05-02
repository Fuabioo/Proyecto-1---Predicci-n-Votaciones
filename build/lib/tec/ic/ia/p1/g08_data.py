from tec.ic.ia.pc1 import g08
import numpy
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


def shaped_data(n,sample_type = 0):
    dataset = numpy.array(g08.generar_muestra_pais(n,sample_type))


    X = dataset[:,1:-2].astype(float)
    X0 = dataset[:,0]
    X32 = dataset[:,-2]
    Y = dataset[:,-1]

    # encode class values as integers
    encoderY = LabelEncoder()
    encoderY.fit(Y)
    encoded_Y = encoderY.transform(Y)

    # encode class values as integers
    encoderX0 = LabelEncoder()
    encoderX0.fit(X0)
    X0 = encoderX0.transform(X0)

    # encode class values as integers
    encoderX32 = LabelEncoder()
    encoderX32.fit(X32)
    X32 = encoderX32.transform(X32)

    X = numpy.concatenate((X0.reshape((-1, 1)), X), axis=1)
    X = numpy.concatenate((X, X32.reshape((-1, 1))), axis=1)

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    return X,dummy_y


def shaped_data_no_bin(n,sample_type = 0):
    dataset = numpy.array(g08.generar_muestra_pais(n,sample_type))


    X = dataset[:,1:-2].astype(float)
    X0 = dataset[:,0]
    X32 = dataset[:,-2]
    X31 = dataset[:,-3]
    
    Y = dataset[:,-1]

    # encode class values as integers
    encoderY = LabelEncoder()
    encoderY.fit(Y)
    encoded_Y = encoderY.transform(Y)

    # encode class values as integers
    encoderX0 = LabelEncoder()
    encoderX0.fit(X0)
    X0 = encoderX0.transform(X0)

    # encode class values as integers
    encoderX32 = LabelEncoder()
    encoderX32.fit(X32)
    X32 = encoderX32.transform(X32)


    encoderX31 = LabelEncoder()
    encoderX31.fit(X31)
    X31 = encoderX32.transform(X31)



    X = numpy.concatenate((X0.reshape((-1, 1)), X), axis=1)


    X = numpy.concatenate((X, X31.reshape((-1, 1))), axis=1)
    X = numpy.concatenate((X, X32.reshape((-1, 1))), axis=1)

    

    Y = numpy.array([g08.PARTIDOS.index(Y[i]) for i in range(len(Y))])

    X = numpy.concatenate((X, Y.reshape((-1, 1))), axis=1)

    return X

shaped_data_no_bin(100,0)


