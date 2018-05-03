import numpy
import pandas
from tec.ic.ia.p1 import g08_data
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split

def baseline_model(y_len):
 	# create model
 	model = Sequential()
 	model.add(Dense(66, input_dim=33, activation='relu'))
 	model.add(Dense(100, input_dim=33, activation='relu'))
 	model.add(Dense(y_len , activation='softmax'))
 	# Compile model
 	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 	return model

def baseline_model2(y_len, hidden_layer_amount, hidden_unit_amount, activation_fun, input_len = 33):

    model = Sequential()
    model.add(Dense(66, input_dim=input_len, activation=activation_fun))
    while hidden_layer_amount > 0:
	    model.add(Dense(hidden_unit_amount, activation=activation_fun))
	    hidden_layer_amount -= 1
    model.add(Dense(y_len, activation='softmax'))
    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


def execute_model(hidden_layer_amount, hidden_unit_amount, activation_fun):

    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    # load dataset

    X, dummy_y = g08_data.shaped_data2(1000)
    '''
    estimator = KerasClassifier(
        build_fn=baseline_model,
        y_len=len(
            dummy_y[0]),
        hidden_layer_amount=hidden_layer_amount,
        hidden_unit_amount=hidden_unit_amount,
        activation_fun=activation_fun,
        epochs=10,
        input_len = len(X[0]),
        batch_size=10,
        verbose=1)
    '''


    estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0, y_len = len(dummy_y[0]))
    X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.33, random_state=seed)
    estimator.fit(X_train, Y_train)
    predictions = estimator.predict(X_test)
    print(predictions)

    print(predictions)






if __name__ == '__main__':
    # execute_model(4, 'relu')
    execute_model(10, 2, 'sigmoid')
