
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pickle

# All Dataset 
class DataSourceReader(object):

    def __init__(self, path):
        df = pd.read_csv(r'{}'.format(path), header=None)
        self.arrayData = df.to_numpy()

    def getAllData(self,filename):
        xArrayData = self.arrayData[:, 0: self.arrayData.shape[1]-1]
        yArrayData = self.arrayData[:,
                                    self.arrayData.shape[1]-1: self.arrayData.shape[1]]
        xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData = train_test_split(
            xArrayData, yArrayData, test_size=0.30)

       
        self.savePikelFile(xTrinArrayData,   "./data/xTrinArrayData"+filename)
        self.savePikelFile(xTestArrayData,   "./data/xTestArrayData"+filename)
        self.savePikelFile(yTrainArrayData,"./data/yTrainArrayData"+filename)
        self.savePikelFile(yTestArrayData,   "./data/yTestArrayData"+filename)

# K-Fold Dataset
    def fold(self, filename):
        kfold = KFold(n_splits=10, shuffle=True, random_state=1)
        # enumerate splits
        xTrain = []
        yTrain = []
        xTest = []
        yTest = []

        for train, test in kfold.split(self.arrayData):
            xTrain.append(
                self.arrayData[train][0:self.arrayData[train].shape[0], 0: self.arrayData[train].shape[1]-1])
            yTrain.append(self.arrayData[train][0:self.arrayData[train].shape[0],
                          self.arrayData[train].shape[1]-1: self.arrayData[train].shape[1]])

            xTest.append(
                self.arrayData[test][0:self.arrayData[test].shape[0], 0: self.arrayData[test].shape[1]-1])
            yTest.append(self.arrayData[test][0:self.arrayData[test].shape[0],
                         self.arrayData[test].shape[1]-1:  self.arrayData[test].shape[1]])
            #print('train: %s, test: %s' % (len(arrayData[train]),len(arrayData[test])))

        #return xTrain, yTrain, xTest, yTest
        self.savePikelFile(xTrain, "./data/xTrain"+ filename)
        self.savePikelFile(yTrain, "./data/yTrain"+ filename)
        self.savePikelFile(xTest,  "./data/xTest"+  filename)
        self.savePikelFile(yTest,  "./data/yTest"+  filename)

        
    def readFile(name):
        myFile= open(name, "rb")
        tempArry =pickle.load(myFile)
        return tempArry

    def savePikelFile(self,array,fileName):
        file_out= open(fileName, "wb")
        pickle.dump(array,file_out)
        file_out.close()






        #return xTrinArrayData, xTestArrayData, yTrainArrayData, yTestArrayData
        # np.savetxt("./data/xTrinArrayData"+filename,xTrinArrayData, delimiter=",")
        # np.savetxt("./data/xTestArrayData"+filename,xTestArrayData, delimiter=",")
        # np.savetxt("./data/yTrainArrayData"+filename, yTrainArrayData, delimiter=",")
        # np.savetxt("./data/yTestArrayData"+filename,yTestArrayData, delimiter=",")
        
        # pd.DataFrame(xTrain.tolist()).to_csv("./data/xTrain"+filename+".csv")
        # pd.DataFrame(yTrain.tolist()).to_csv("./data/yTrain"+filename+".csv")
        # pd.DataFrame(xTest.tolist()).to_csv("./data/xTest"+filename+".csv")
        # pd.DataFrame(yTest.tolist()).to_csv("./data/yTest"+filename+".csv")

        # np.savetxt("./data/xTrain"+filename, np.array(xTrain).reshape(-1, 3), delimiter="," , fmt='%d')
        # np.savetxt("./data/yTrain"+filename, np.array(yTrain).reshape(-1, 3), delimiter=",", fmt='%d')
        # np.savetxt("./data/xTest"+filename, np.array(xTest).reshape(-1, 3), delimiter=",", fmt='%d')
        # np.savetxt("./data/yTest"+filename, np.array(yTest).reshape(-1, 3), delimiter=",", fmt='%d')


        #return xTrain, yTrain, xTest, yTest
        
        # df=pd.DataFrame(xTrain)
        # df.to_csv("./data/xTrain.csv", header=False)
        # pd.DataFrame(yTrain).to_csv("./data/yTrain.csv")
        # pd.DataFrame(xTest).to_csv("./data/xTest.csv")
        # pd.DataFrame(yTest).to_csv("./data/yTest.csv")
