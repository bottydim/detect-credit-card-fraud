import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import Model
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils import get_engine
from sklearn import linear_model

if __name__ == '__main__':

    # load dataset
    table = "data_little_event"
    engine = get_engine()
    dataframe = pd.read_sql_query("select * from {table}".format(table=table),engine)
    dataset = dataframe.values
    # print("First one row of the dataset")
    # print(dataset[0:2,:])

    # split into input (X) and output (Y) variables
    data_dimensions = 64
    X = dataset[:, 1:data_dimensions]
    Y = dataset[:, data_dimensions]
    # Y = to_categorical(Y, nb_classes=None)


    input_dimensions = X.shape[1]




    print("shapes: X[{}]=====Y[{}]".format(X.shape,Y.shape))
    print("Fraud {}% ".format(float(np.sum(Y==1))*100.0/Y.shape[0]))
    print("Total #samples:",Y.shape[0])



    # define base mode
    def baseline_model():
        return logistic_regresion()
        # return linear_regression()


    def keras_lin_reg():
        x = Input((None,input_dimensions))
        y = Dense(1,activation='linear')(x)
        model = Model(x,y,"Linear Regression")
        model.compile(loss='mse', optimizer='sgd')
        return model

    def logistic_regresion():
        logistic = linear_model.LogisticRegression(solver='sag', n_jobs=-1,max_iter=500)
        return logistic
    def linear_regression():
        lr = linear_model.LinearRegression(n_jobs=-1)
        return lr

    def mlp_model():
        # create model
        model = Sequential()
        model.add(Dense(input_dimensions, input_dim=input_dimensions, init='normal', activation='relu'))
        model.add(Dense(2, init='normal', activation='softmax'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # evaluate model with standardized dataset
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    # estimators.append(('mlp', KerasClassifier(build_fn=baseline_model, nb_epoch=100, batch_size=10000, verbose=1)))
    estimators.append(('liner reg', KerasClassifier(build_fn=keras_lin_reg, nb_epoch=100, batch_size=100000, verbose=1)))
    # estimators.append(('linear_reg', baseline_model()))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(pipeline, X, Y, cv=kfold, scoring='roc_auc')
    print("Results:", results)
    print("Results: %.24f (%.24f) ROC" % (results.mean(), results.std()))
    print(pipeline)