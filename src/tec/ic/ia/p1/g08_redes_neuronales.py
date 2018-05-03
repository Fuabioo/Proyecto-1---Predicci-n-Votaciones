import numpy
import pandas
from tec.ic.ia.p1 import g08_data
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold



def baseline_model(y_len,x_len):
    # create model
    model = Sequential()
    model.add(Dense(x_len*4, input_dim=x_len, activation='relu'))
    model.add(Dense(x_len*3, activation='relu'))
    model.add(Dense(x_len*2, activation='relu'))
    model.add(Dense(x_len,   activation='relu'))
    model.add(Dense(y_len , activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def baseline_model2(y_len, hidden_layer_amount, hidden_unit_amount, activation_fun):

    model = Sequential()
    model.add(Dense(66, input_dim=33, activation=activation_fun))
    while hidden_layer_amount > 0:
	    model.add(Dense(hidden_unit_amount, activation=activation_fun))
	    hidden_layer_amount -= 1
    model.add(Dense(y_len, activation='softmax'))
    # Compile model
    model.compile(
        loss='_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


def execute_model(hidden_layer_amount, hidden_unit_amount, activation_fun):

    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    # load dataset

    [X1, Y1],[X2, Y2],[X3, Y3] = g08_data.shaped_data2(10000)
    X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=0.20, random_state=seed)

    cvmodels = []
    cvscores = []
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for train, test in kfold.split(X_train, Y_train):


        dummy_y = np_utils.to_categorical(Y_train)

        estimator = baseline_model(len(dummy_y[0]),len(X_train[0]))

        estimator.fit(X_train[train], dummy_y[train], epochs=100, batch_size=100, verbose=0)

        scores = estimator.evaluate(X_train[test], dummy_y[test], verbose=0)

        cvscores.append(scores[1] * 100)
        cvmodels.append(estimator)



    estimator = cvmodels[cvscores.index(max(cvscores))]

    predictions = estimator.predict_classes(X_test)

    success = 0
    for i in range(len(predictions)):
        if predictions[i] == Y_test[i]:
            success+=1
    print(110*success/len(predictions))
    partidos1 = [g08.PARTIDOS[int(predictions[i])] for i in range(len(predictions))]






    X_train, X_test, Y_train, Y_test = train_test_split(X2, Y2, test_size=0.20, random_state=seed)

    cvmodels = []
    cvscores = []
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for train, test in kfold.split(X_train, Y_train):


        dummy_y = np_utils.to_categorical(Y_train)

        estimator = baseline_model(len(dummy_y[0]),len(X_train[0]))

        estimator.fit(X_train[train], dummy_y[train], epochs=100, batch_size=100, verbose=0)

        scores = estimator.evaluate(X_train[test], dummy_y[test], verbose=0)

        cvscores.append(scores[1] * 100)
        cvmodels.append(estimator)



    estimator = cvmodels[cvscores.index(max(cvscores))]

    predictions = estimator.predict_classes(X_test)

    success = 0
    for i in range(len(predictions)):
        if predictions[i] == Y_test[i]:
            success+=1
    print(110*success/len(predictions))
    partidos2 = [g08.PARTIDOS[int(predictions[i])] for i in range(len(predictions))]


    X_train, X_test, Y_train, Y_test = train_test_split(X3, Y3, test_size=0.20, random_state=seed)

    cvmodels = []
    cvscores = []
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for train, test in kfold.split(X_train, Y_train):


        dummy_y = np_utils.to_categorical(Y_train)

        estimator = baseline_model(len(dummy_y[0]),len(X_train[0]))

        estimator.fit(X_train[train], dummy_y[train], epochs=100, batch_size=100, verbose=0)

        scores = estimator.evaluate(X_train[test], dummy_y[test], verbose=0)

        cvscores.append(scores[1] * 100)
        cvmodels.append(estimator)



    estimator = cvmodels[cvscores.index(max(cvscores))]

    predictions = estimator.predict_classes(X_test)

    success = 0
    for i in range(len(predictions)):
        if predictions[i] == Y_test[i]:
            success+=1
    print(110*success/len(predictions))
    partidos3 = [g08.PARTIDOS[int(predictions[i])] for i in range(len(predictions))]


    return X1,partidos1,partidos2,partidos3



if __name__ == '__main__':
    # execute_model(4, 'relu')
    execute_model(10, 2, 'sigmoid')
