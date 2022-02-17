from keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from keras.layers import Activation
from keras.layers.core import Dense
from sklearn.cluster import KMeans
from keras.initializers import Initializer
from sklearn.metrics import f1_score
from keras.initializers import RandomUniform, Initializer, Constant
from keras.utils.layer_utils import get_source_inputs
from tensorflow.python.keras.layers import Layer
from keras import backend as K
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from sklearn.model_selection import PredefinedSplit
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

from libs.initCentersKMeans import InitCentersKMeans
from libs.rbfLayer import RBFLayer
from libs.utils import Utils
from libs.dataSourceReader import DataSourceReader
from numpy import genfromtxt

import traceback
from libs.PPCA import PPCA

class Calculation(object):
    def __init__(self, args):
        self.args = args

    def calculate(self, fileName):
            
        xTrinArrayData  = DataSourceReader.readFile("./data/xTrinArrayData"+fileName)
        xTestArrayData  = DataSourceReader.readFile("./data/xTestArrayData"+fileName)
        yTrainArrayData = DataSourceReader.readFile("./data/yTrainArrayData"+fileName)
        yTestArrayData  = DataSourceReader.readFile("./data/yTestArrayData"+fileName)

        xTrain = DataSourceReader.readFile("./data/xTrain"+fileName)
        yTrain = DataSourceReader.readFile("./data/yTrain"+fileName)
        xTest  = DataSourceReader.readFile("./data/xTest"+fileName)
        yTest  = DataSourceReader.readFile("./data/yTest"+fileName)
        
    # ############## Call Function
        #baseline
        if self.args.i == '0' or self.args.i == None:
            try:
                self.KNN_on_BASELINE(xTrain, yTrain, xTest, yTest)
            except Exception as e:
                print("KNN_on_BASELINE had error: " + str(e) + ' : ' + traceback.format_exc())
        
        if self.args.i == '1' or self.args.i == None:
            try:
                self.RF_on_BASELINE(xTrain, yTrain, xTest, yTest)
            except Exception as e:
                print("RF_on_BASELINE had error: " + str(e))

        if self.args.i == '2' or self.args.i == None:
            try:
                self.MLP_on_BASELINE(xTrain, yTrain, xTest, yTest)
            except Exception as e:
                print("MLP_on_BASELINE had error: " + str(e))

        if self.args.i == '3' or self.args.i == None:
            try:
                self.RBF_on_BASELINE(xTrain, yTrain, xTest, yTest)
            except Exception as e:
                print("RBF_on_BASELINE had error: " + str(e) + traceback.format_exc())

        #lle_dr
        if self.args.i == '4' or self.args.i == None:
            try:
                self.LLE_KNN_on_DR(xTrain, yTrain, xTest, yTest)
            except Exception as e:
                print("LLE_KNN_on_DR had error: " + str(e))

        if self.args.i == '5' or self.args.i == None:
            try:
                self.LLE_RF_on_DR(xTrain, yTrain, xTest, yTest)
            except Exception as e:
                print("LLE_RF_on_DR had error: " + str(e))

        if self.args.i == '6' or self.args.i == None:
            try:
                self.LLE_MLP_on_DR(xTrain, yTrain, xTest, yTest)
            except Exception as e:
                print("LLE_MLP_on_DR had error: " + str(e))

        if self.args.i == '7' or self.args.i == None:
            try:
                self.LLE_RBF_on_DR(xTrain, yTrain, xTest, yTest)
            except Exception as e:
                print("LLE_RBF_on_DR had error: " + str(e))

        #lle_var
        if self.args.i == '8' or self.args.i == None:
            try:
                self.LLE_KNN_explaindvarince(xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData)
            except Exception as e:
                print("LLE_KNN_on_VAR had error: " + str(e))

        if self.args.i == '9' or self.args.i == None:
            try:
                self.LLE_RF_explaindvarince(xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData)
            except Exception as e:
                print("LLE_RF_on_VAR had error: " + str(e))

        if self.args.i == '10' or self.args.i == None:
            try:
                self.LLE_MLP_explaindvarince(xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData)
            except Exception as e:
                print("LLE_MLP_on_VAR had error: " + str(e))

        if self.args.i == '11' or self.args.i == None:
            try:
                self.LLE_RBF_explaindvarince(xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData)
            except Exception as e:
                print("LLE_RBF_on_VAR had error: " + str(e))

        #ppca_var
        if self.args.i == '12' or self.args.i == None:
            try:
                self.PPCA_KNN_explaindvarince(xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData)
            except Exception as e:
                print("PPCA_KNN_on_VAR had error: " + str(e))
        
        if self.args.i == '13' or self.args.i == None:
            try:
                self.PPCA_RF_explaindvarince(xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData)
            except Exception as e:
                print("PPCA_RF_on_VAR had error: " + str(e))
   
        if self.args.i == '14' or self.args.i == None:
            try:
                self.PPCA_MLP_explaindvarince(xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData)
            except Exception as e:
                print("PPCA_MLP_on_VAR had error: " + str(e))

        if self.args.i == '15' or self.args.i == None:
            try:
                self.PPCA_RBF_explaindvarince(xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData)
            except Exception as e:
                print("PPCA_RBF_on_VAR had error: " + str(e))

         #lle_dr
       
        #ppca_dr
        if self.args.i == '16' or self.args.i == None:
            try:
                self.PPCA_KNN_on_DR(xTrain, yTrain, xTest, yTest)
            except Exception as e:
                print("PPCA_KNN_on_DR had error: " + str(e))

        if self.args.i == '17' or self.args.i == None:
            try:
                self.PPCA_RF_on_DR(xTrain, yTrain, xTest, yTest)
            except Exception as e:
                print("PPCA_RF_on_DR had error: " + str(e))

        if self.args.i == '18' or self.args.i == None:
            try:
                self.PPCA_MLP_on_DR(xTrain, yTrain, xTest, yTest)
            except Exception as e:
                print("PPCA_MLP_on_DR had error: " + str(e))

        if self.args.i == '19' or self.args.i == None:
            try:
                self.PPCA_RBF_on_DR(xTrain, yTrain, xTest, yTest)
            except Exception as e:
                print("PPCA_RBF_on_DR had error: " + str(e))

    # ############## Calculate Variance
    def variance(self, mainData, data):
        mainVar= np.var(mainData)
        
        for i in range(data.shape[1]):
            var = np.var(data[:,i])
            ratio = (var/mainVar) #/ 1000
            print(" col number , ratio , mainVar , var  ", [ i ,  ratio , mainVar ,var] )
            print("#####################################")

 
    # ############## VAR#####################################################################
    #LLE_VAR*********************************************************************************
    def LLE_KNN_on_VAR(self, xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData):
        
        result=[]
        accuracy = 0
        f1 = 0
        step = True
        number = 2
        while(number<=220):
            embedding = LocallyLinearEmbedding(
                n_components=number, n_neighbors=10, eigen_solver="dense", method='standard')
            xTrainTransformed = embedding.fit_transform(xTrinArrayData)
            xTestTransformed = embedding.fit_transform(xTestArrayData)
            F1score, accuracyscore = Utils.calcKNN(
                xTrainTransformed, yTrainArrayData,  xTestTransformed, yTestArrayData)
            
            #if(accuracyscore-accuracy <= 0):  # or F1score- f1 <0.001):
            print("Number of feature , F1score , accuracyscore ", [number , F1score , accuracyscore])
            result.append([number , F1score, accuracyscore, xTrainTransformed] )
            number = number+2
            
        return result
    
    def LLE_RF_on_VAR(self, xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData):
        
        result=[]
        accuracy = 0
        f1 = 0
        step = True
        number = 2
        while(number<=220):
            embedding = LocallyLinearEmbedding(
                n_components=number, n_neighbors=10, eigen_solver="dense")
            xTrainTransformed = embedding.fit_transform(xTrinArrayData)
            xTestTransformed = embedding.fit_transform(xTestArrayData)
            F1score, accuracyscore = Utils.calcRF(
                xTrainTransformed, yTrainArrayData,  xTestTransformed, yTestArrayData)
            
            #if(accuracyscore-accuracy <= 0):  # or F1score- f1 <0.001):
            print("Number of feature , F1score , accuracyscore ", [number , F1score , accuracyscore])
            result.append([number , F1score, accuracyscore, xTrainTransformed] )
            number = number+2
            
        return result
   
    def LLE_MLP_on_VAR(self, xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData):
        
        result=[]
        accuracy = 0
        f1 = 0
        step = True
        number = 2
        while(number<=3):
            embedding = LocallyLinearEmbedding(
                n_components=number, n_neighbors=5, eigen_solver="dense")
            xTrainTransformed = embedding.fit_transform(xTrinArrayData)
            xTestTransformed = embedding.fit_transform(xTestArrayData)
            F1score, accuracyscore = Utils.calcMLP(
                xTrainTransformed, yTrainArrayData,  xTestTransformed, yTestArrayData)
            
            #if(accuracyscore-accuracy <= 0):  # or F1score- f1 <0.001):
            print("Number of feature , F1score , accuracyscore ", [number , F1score , accuracyscore])
            result.append([number , F1score, accuracyscore, xTrainTransformed] )
            number = number+1
            
        return result
    
    def LLE_RBF_on_VAR(self, xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData):
        
        result=[]
        accuracy = 0
        f1 = 0
        step = True
        number = 2
        while(number<=3):
            embedding = LocallyLinearEmbedding(
                n_components=number, n_neighbors=5, eigen_solver="dense")
            xTrainTransformed = embedding.fit_transform(xTrinArrayData)
            xTestTransformed = embedding.fit_transform(xTestArrayData)
            F1score, accuracyscore = Utils.calcRBF(
                xTrainTransformed, yTrainArrayData,  xTestTransformed, yTestArrayData)
            
            #if(accuracyscore-accuracy <= 0):  # or F1score- f1 <0.001):
            print("Number of feature , F1score , accuracyscore ", [number , F1score , accuracyscore])
            result.append([number , F1score, accuracyscore, xTrainTransformed] )
            number = number+1
            
        return result


    def LLE_KNN_explaindvarince(self, xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData):
        tempaar=self.LLE_KNN_on_VAR(xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData)
        arrresult=np.array(tempaar)
        maxval=arrresult[0:arrresult.shape[0],2].max( axis=0)
        index= np.where(arrresult[0:arrresult.shape[0],2] == maxval)
        subData =arrresult[index,3]
        fulldata= np.concatenate((xTrinArrayData, xTestArrayData), axis=0)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Best performance data :Number of feature , F1score , accuracyscore "
        , [arrresult[index,0] , arrresult[index,1] , arrresult[index,2]])
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        self.variance(fulldata,subData[0][0])
    
    def LLE_RF_explaindvarince(self, xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData):
        tempaar=self.LLE_RF_on_VAR(xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData)
        arrresult=np.array(tempaar)
        maxval=arrresult[0:arrresult.shape[0],2].max( axis=0)
        index= np.where(arrresult[0:arrresult.shape[0],2] == maxval)
        subData =arrresult[index,3]
        fulldata= np.concatenate((xTrinArrayData, xTestArrayData), axis=0)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Best performance data :Number of feature , F1score , accuracyscore "
        , [arrresult[index,0] , arrresult[index,1] , arrresult[index,2]])
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        self.variance(fulldata,subData[0][0])

    def LLE_MLP_explaindvarince(self, xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData):
        tempaar=self.LLE_MLP_on_VAR(xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData)
        arrresult=np.array(tempaar)
        maxval=arrresult[0:arrresult.shape[0],2].max( axis=0)
        index= np.where(arrresult[0:arrresult.shape[0],2] == maxval)
        subData =arrresult[index,3]
        fulldata= np.concatenate((xTrinArrayData, xTestArrayData), axis=0)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Best performance data :Number of feature , F1score , accuracyscore "
        , [arrresult[index,0] , arrresult[index,1] , arrresult[index,2]])
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        self.variance(fulldata,subData[0][0])
    
    def LLE_RBF_explaindvarince(self, xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData):
        tempaar=self.LLE_RBF_on_VAR(xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData)
        arrresult=np.array(tempaar)
        maxval=arrresult[0:arrresult.shape[0],2].max( axis=0)
        index= np.where(arrresult[0:arrresult.shape[0],2] == maxval)
        subData =arrresult[index,3]
        fulldata= np.concatenate((xTrinArrayData, xTestArrayData), axis=0)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Best performance data :Number of feature , F1score , accuracyscore "
        , [arrresult[index,0] , arrresult[index,1] , arrresult[index,2]])
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        self.variance(fulldata,subData[0][0])


    #PPCA_VAR*********************************************************************************
    def PPCA_KNN_on_VAR(self, xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData):
        
        result=[]
        accuracy = 0
        f1 = 0
        step = True
        number = 2
        while(number<=100):
            
            ppca = PPCA()
            ppca.fit(data=xTrinArrayData, d=number, verbose=True)
            xTrainTransformed = ppca.transform()
            ppca.fit(data=xTestArrayData, d=number, verbose=True)
            xTestTransformed = ppca.transform()
            F1score, accuracyscore = Utils.calcKNN(xTrainTransformed, yTrainArrayData,  xTestTransformed,  yTestArrayData)
            #if(accuracyscore-accuracy <= 0):  # or F1score- f1 <0.001):
            print("Number of feature , F1score , accuracyscore ", [number , F1score , accuracyscore])
            result.append([number , F1score, accuracyscore, xTrainTransformed] )
            number = number+10
            
        return result

    def PPCA_RF_on_VAR(self, xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData):
        
        result=[]
        accuracy = 0
        f1 = 0
        step = True
        number = 2
        while(number<=50):
            ppca = PPCA()
            ppca.fit(data=xTrinArrayData, d=number, verbose=True)
            xTrainTransformed = ppca.transform()
            ppca.fit(data=xTestArrayData, d=number, verbose=True)
            xTestTransformed = ppca.transform()
            F1score, accuracyscore = Utils.calcRF(xTrainTransformed, yTrainArrayData,  xTestTransformed,  yTestArrayData)
            #if(accuracyscore-accuracy <= 0):  # or F1score- f1 <0.001):
            print("Number of feature , F1score , accuracyscore ", [number , F1score , accuracyscore])
            result.append([number , F1score, accuracyscore, xTrainTransformed] )
            number = number+5
            
        return result
   
    def PPCA_MLP_on_VAR(self, xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData):
        
        result=[]
        accuracy = 0
        f1 = 0
        step = True
        number = 2
        while(number<=200):
            ppca = PPCA()
            ppca.fit(data=xTrinArrayData, d=number, verbose=True)
            xTrainTransformed = ppca.transform()
            ppca.fit(data=xTestArrayData, d=number, verbose=True)
            xTestTransformed = ppca.transform()
            F1score, accuracyscore = Utils.calcMLP(xTrainTransformed, yTrainArrayData,  xTestTransformed,  yTestArrayData)
            #if(accuracyscore-accuracy <= 0):  # or F1score- f1 <0.001):
            print("Number of feature , F1score , accuracyscore ", [number , F1score , accuracyscore])
            result.append([number , F1score, accuracyscore, xTrainTransformed] )
            number = number+2
            
        return result
    
    def PPCA_RBF_on_VAR(self, xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData):
        
        result=[]
        accuracy = 0
        f1 = 0
        step = True
        number = 16
        while(number<=50):
            ppca = PPCA()
            ppca.fit(data=xTrinArrayData, d=number, verbose=True)
            xTrainTransformed = ppca.transform()
            ppca.fit(data=xTestArrayData, d=number, verbose=True)
            xTestTransformed = ppca.transform()
            F1score, accuracyscore = Utils.calcRBF(xTrainTransformed, yTrainArrayData,  xTestTransformed,  yTestArrayData)
            #if(accuracyscore-accuracy <= 0):  # or F1score- f1 <0.001):
            print("Number of feature , F1score , accuracyscore ", [number , F1score , accuracyscore])
            result.append([number , F1score, accuracyscore, xTrainTransformed] )
            number = number+5
            
        return result


    def PPCA_KNN_explaindvarince(self, xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData):
        
        tempaar=self.PPCA_KNN_on_VAR(xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData)
        arrresult=np.array(tempaar)
        maxval=arrresult[0:arrresult.shape[0],2].max( axis=0)
        index= np.where(arrresult[0:arrresult.shape[0],2] == maxval)
        subData =arrresult[index,3]
        fulldata= np.concatenate((xTrinArrayData, xTestArrayData), axis=0)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Best performance data :Number of feature , F1score , accuracyscore "
        , [arrresult[index,0] , arrresult[index,1] , arrresult[index,2]])
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        self.variance(fulldata,subData[0][0])

    def PPCA_RF_explaindvarince(self, xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData):
        tempaar=self.PPCA_RF_on_VAR(xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData)
        arrresult=np.array(tempaar)
        maxval=arrresult[0:arrresult.shape[0],2].max( axis=0)
        index= np.where(arrresult[0:arrresult.shape[0],2] == maxval)
        subData =arrresult[index,3]
        fulldata= np.concatenate((xTrinArrayData, xTestArrayData), axis=0)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Best performance data :Number of feature , F1score , accuracyscore "
        , [arrresult[index,0] , arrresult[index,1] , arrresult[index,2]])
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        self.variance(fulldata,subData[0][0])

    def PPCA_MLP_explaindvarince(self, xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData):
        tempaar=self.PPCA_MLP_on_VAR(xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData)
        arrresult=np.array(tempaar)
        maxval=arrresult[0:arrresult.shape[0],2].max( axis=0)
        index= np.where(arrresult[0:arrresult.shape[0],2] == maxval)
        subData =arrresult[index,3]
        fulldata= np.concatenate((xTrinArrayData, xTestArrayData), axis=0)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Best performance data :Number of feature , F1score , accuracyscore "
        , [arrresult[index,0] , arrresult[index,1] , arrresult[index,2]])
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        self.variance(fulldata,subData[0][0])
    
    def PPCA_RBF_explaindvarince(self, xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData):
        tempaar=self.PPCA_RBF_on_VAR(xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData)
        arrresult=np.array(tempaar)
        maxval=arrresult[0:arrresult.shape[0],2].max( axis=0)
        index= np.where(arrresult[0:arrresult.shape[0],2] == maxval)
        subData =arrresult[index,3]
        fulldata= np.concatenate((xTrinArrayData, xTestArrayData), axis=0)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Best performance data :Number of feature , F1score , accuracyscore "
        , [arrresult[index,0] , arrresult[index,1] , arrresult[index,2]])
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        self.variance(fulldata,subData[0][0])

  
    # ############## BASELINE###############################################################
    def KNN_on_BASELINE(self, xTrain, yTrain, xTest, yTest):
        
        for i in range(10):
            print("Values for fold ",  i+1)
            Utils.calcKNN(xTrain[i],  np.ravel(yTrain[i]),
                        xTest[i],  np.ravel(yTest[i]))
            print("################################")
            #x = np.linspace(0, 20, 100)
            #plt.plot(x, np.sin(x))
            #plt.show()            

    def RF_on_BASELINE(self, xTrain, yTrain, xTest, yTest):
        
        for i in range(10):
            print("Values for fold ",  i+1)
            Utils.calcRF(xTrain[i],  np.ravel(yTrain[i]),
                        xTest[i],  np.ravel(yTest[i]))

    def MLP_on_BASELINE(self, xTrain, yTrain, xTest, yTest):
        
        for i in range(10):
            print("Values for fold ",  i+1)
            Utils.calcMLP(xTrain[i],  np.ravel(yTrain[i]),
                        xTest[i],  np.ravel(yTest[i]))
 
    def RBF_on_BASELINE(self, xTrain, yTrain, xTest, yTest):
        
        for i in range(10):
            print("Values for fold ",  i+1)
            Utils.calcRBF(xTrain[i],  np.ravel(yTrain[i]),
                        xTest[i],  np.ravel(yTest[i]))

 
   # ############## DIMENSION REDUCTION####################################################
    # LLE_DR********************************************************************************
    def LLE_KNN_on_DR(self, xTrain, yTrain, xTest, yTest):
       
        number = 16
        for i in range(10):
            print("Values for fold ",  i+1)
            embedding = LocallyLinearEmbedding(
                n_components=number, n_neighbors=10, eigen_solver="dense")
            xTrainTransformed = embedding.fit_transform(xTrain[i])
            xTestTransformed = embedding.fit_transform(xTest[i])
            Utils.calcKNN(xTrainTransformed, yTrain[i],  xTestTransformed, yTest[i])
            F1score, accuracyscore = Utils.calcKNN(xTrainTransformed, yTrain[i],  xTestTransformed,  yTest[i])
            print("F1score:", F1score)
            print("accuracyscore:", accuracyscore)

    def LLE_RF_on_DR(self, xTrain, yTrain, xTest, yTest):
       
        number = 68
        for i in range(10):
            print("Values for fold ",  i+1)
            embedding = LocallyLinearEmbedding(
                n_components=number, n_neighbors=10, eigen_solver="dense")
            xTrainTransformed = embedding.fit_transform(xTrain[i])
            xTestTransformed = embedding.fit_transform(xTest[i])
            F1score, accuracyscore = Utils.calcRF(xTrainTransformed, yTrain[i],  xTestTransformed, yTest[i])
            print("F1score:", F1score)
            print("accuracyscore:", accuracyscore)
    
    def LLE_MLP_on_DR(self, xTrain, yTrain, xTest, yTest):
        
        number = 2
        for i in range(10):
            print("Values for fold ",  i+1)
            embedding = LocallyLinearEmbedding(
                n_components=number, n_neighbors=5, eigen_solver="dense")
            xTrainTransformed = embedding.fit_transform(xTrain[i])
            xTestTransformed = embedding.fit_transform(xTest[i])
            F1score, accuracyscore = Utils.calcMLP(xTrainTransformed, yTrain[i],  xTestTransformed, yTest[i])
            print("F1score:", F1score)
            print("accuracyscore:", accuracyscore)

    def LLE_RBF_on_DR(self, xTrain, yTrain, xTest, yTest):
      
        number = 2
        for i in range(10):
            print("Values for fold ",  i+1)
            embedding = LocallyLinearEmbedding(
                n_components=number, n_neighbors=5, eigen_solver="dense")
            xTrainTransformed = embedding.fit_transform(xTrain[i])
            xTestTransformed = embedding.fit_transform(xTest[i])
            F1score, accuracyscore = Utils.calcRBF(xTrainTransformed, yTrain[i],  xTestTransformed, yTest[i])
            print("F1score:", F1score)
            print("accuracyscore:", accuracyscore)
    
    # PPCA_DR********************************************************************************
    def PPCA_KNN_on_DR(self, xTrain, yTrain, xTest, yTest):
       
        #LLE - KNN
        number = 32
        for i in range(10): 
            print("Values for fold ",  i+1)
            ppca = PPCA()
            ppca.fit(data=xTrain[i], d=number, verbose=True)
            xTrainTransformed = ppca.transform()
            ppca.fit(data=xTest[i], d=number, verbose=True)
            xTestTransformed = ppca.transform()
            F1score, accuracyscore = Utils.calcKNN(xTrainTransformed, yTrain[i],  xTestTransformed,  yTest[i])
            print("F1score:", F1score)
            print("accuracyscore:", accuracyscore)

    def PPCA_RF_on_DR(self, xTrain, yTrain, xTest, yTest):
        
        #LLE - KNN
        number = 36
        for i in range(10): 
            print("Values for fold ",  i+1)
            ppca = PPCA()
            ppca.fit(data=xTrain[i], d=number, verbose=True)
            xTrainTransformed = ppca.transform()
            ppca.fit(data=xTest[i], d=number, verbose=True)
            xTestTransformed = ppca.transform()
            F1score, accuracyscore = Utils.calcRF(xTrainTransformed, yTrain[i],  xTestTransformed,  yTest[i])
            print("F1score:", F1score)
            print("accuracyscore:", accuracyscore)

    def PPCA_MLP_on_DR(self, xTrain, yTrain, xTest, yTest):
       
        #LLE - KNN
        number = 7
        for i in range(10): 
            print("Values for fold ",  i+1)
            ppca = PPCA()
            ppca.fit(data=xTrain[i], d=number, verbose=True)
            xTrainTransformed = ppca.transform()
            ppca.fit(data=xTest[i], d=number, verbose=True)
            xTestTransformed = ppca.transform()
            F1score, accuracyscore = Utils.calcMLP(xTrainTransformed, yTrain[i],  xTestTransformed,  yTest[i])
            print("F1score:", F1score)
            print("accuracyscore:", accuracyscore)

    def PPCA_RBF_on_DR(self, xTrain, yTrain, xTest, yTest):
       
        #LLE - KNN
        number = 2
        for i in range(10): 
            print("Values for fold ",  i+1)
            ppca = PPCA()
            ppca.fit(data=xTrain[i], d=number, verbose=True)
            xTrainTransformed = ppca.transform()
            ppca.fit(data=xTest[i], d=number, verbose=True)
            xTestTransformed = ppca.transform()
            F1score, accuracyscore = Utils.calcRBF(xTrainTransformed, yTrain[i],  xTestTransformed,  yTest[i])
            print("F1score:", F1score)
            print("accuracyscore:", accuracyscore)


    # LIM_DR********************************************************************************
    # Octave_Software

  #xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData = dataSource.getAllData( fileName)
        #xTrain, yTrain, xTest, yTest = dataSource.fold()
        