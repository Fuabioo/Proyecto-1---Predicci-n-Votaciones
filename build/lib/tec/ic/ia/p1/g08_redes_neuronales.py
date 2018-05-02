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


# define baseline model
# def baseline_model(y_len):
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(66, input_dim=33, activation='relu'))
# 	model.add(Dense(100, input_dim=33, activation='relu'))
# 	model.add(Dense(y_len , activation='softmax'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 	return model

def baseline_model(y_len, hidden_layer_amount, hidden_unit_amount, activation_fun):

    model = Sequential()
    model.add(Dense(66, input_dim=33, activation=activation_fun))
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

    X, dummy_y = g08_data.shaped_data(10000)

    estimator = KerasClassifier(
        build_fn=baseline_model,
        y_len=len(
            dummy_y[0]),
        hidden_layer_amount=hidden_layer_amount,
        hidden_unit_amount=hidden_unit_amount,
        activation_fun=activation_fun,
        epochs=100,
        batch_size=100,
        verbose=2)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    results = cross_val_score(estimator, X, dummy_y, cv=kfold)

    print("Baseline: %.2f%% (%.2f%%)" %
          (results.mean() * 100, results.std() * 100))


if __name__ == '__main__':
    # execute_model(4, 'relu')
    execute_model(10, 2, 'sigmoid')
