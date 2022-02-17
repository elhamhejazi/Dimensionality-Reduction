from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from libs.rbfLayer import RBFLayer
from libs.initCentersKMeans import InitCentersKMeans
from keras.layers.core import Dense
from keras.layers import Activation
from tensorflow.keras.optimizers import RMSprop
import tensorflow_addons as tfa
import numpy as np


class Utils(object):
    @staticmethod
    def trainModel(Method, Param, xTrain, yTrain):
        #p= PredefinedSplit(test_fold=xTrain([ 0,  1]))
        tempClf = GridSearchCV(Method, Param, cv=5,
                               refit=True, scoring='accuracy')
        tempClf.fit(xTrain, np.ravel(yTrain))
        print("Clf", tempClf.best_params_)
        return tempClf

    @staticmethod
    def calcPerformance(clfTune, x, y):
        trainYPred = clfTune.predict(x)
        F1score = f1_score(trainYPred, np.ravel(y), average='macro')
        accuracyscore = accuracy_score(trainYPred, np.ravel(y))
        print("f1Score Performance:", F1score)
        print("accuracy Score Performance:", accuracyscore)
        return F1score, accuracyscore

    @staticmethod
    def calcKNN(x, y, xt, yt):
        model = KNeighborsClassifier()
        param = [{'n_neighbors': [3,5,10,15,50,100]}]
        clf = Utils.trainModel(model, param, x, y)
        return Utils.calcPerformance(clf, xt, yt)

    @staticmethod
    def calcRF(x, y, xt, yt):
        model = RandomForestClassifier()
        param = [{'max_depth': [20,50,100,200,300], 'random_state':[0, 1, 2]}]
        clf = Utils.trainModel(model, param, x, y)
        return Utils.calcPerformance(clf, xt, yt)

    @staticmethod
    def calcMLP(x, y, xt, yt):
        model = MLPClassifier()
        # , 'random_state':[1]}
        #clf = MLPClassifier(hidden_layer_sizes=100,random_state=2,alpha=0.0001,
        #batch_size=60,tol=0.0005,activation='identity',
        #learning_rate_init=0.01,max_iter=1000)
        param = [{'max_iter': [20,100,200,500], 'random_state':[0,1]}]#,'hidden_layer_sizes':[100]
        clf = Utils.trainModel(model, param, x, y)
        return Utils.calcPerformance(clf, xt, yt)

    @staticmethod
    def calcRBF(x, y, xt, yt):
        X = MinMaxScaler().fit_transform(x)
        ohe = OneHotEncoder()
        Y = ohe.fit_transform(y.reshape(-1, 1)).toarray()
        
        #Dense D1 To D12 =70 , D13=10 
        model = Sequential()
        rbflayer = RBFLayer(200, initializer=InitCentersKMeans(
            X), betas=3.0, input_shape=(354,))
        model.add(rbflayer)
        model.add(Dense(70))
        model.add(Activation('linear'))
        model.compile(loss='mean_squared_error', optimizer=RMSprop(),
                      metrics=['accuracy', tfa.metrics.F1Score(num_classes=2, average="micro")])
        history1 = model.fit(X, Y, epochs=10, batch_size=32)

        return history1
