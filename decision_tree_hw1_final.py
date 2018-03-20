import typing
import math
import operator
import numpy as np
from scipy import stats
from pprint import pprint
import re

class DecisionTreeHW(object):

    def __init__(self) -> None:
        print('ID3 under way...')

    def attSplit(self, data: list) -> dict:
       
            return {val: (data==val).nonzero()[0] for val in np.unique(data)}
        
    def entropy(self, data: list) -> float:
        res = 0
        val, counts = np.unique(data, return_counts=True)
        freqs = counts.astype('float')/len(data)
        for p in freqs:
            if p != 0.0:
                res -= p * np.log2(p)
        return res
        
    def informationGain(self, y: dict, x: dict) -> float:

        res = self.entropy(y)

        # We partition x, according to attribute values x_i
        val, counts = np.unique(x, return_counts=True)
        freqs = counts.astype('float')/len(x)

        # We calculate a weighted average of the entropy
        for p, v in zip(freqs, val):
            res -= p * self.entropy(y[x == v])

        return res

    def uniform(self, data: dict):
        return len(set(data)) == 1

    def train(self, x: dict, y: dict, depth: int = None, currentDepth: int = -1):

         
        """        
        if depth:
            d = depth-1
        else:
            d = 0
        """    
        if self.uniform(y) or len(y) == 0 or depth == currentDepth:
            return stats.mode(y)[0]
        
        gain = np.array([self.informationGain(y, x_attr) for x_attr in x.T])
        
        selected_attr = np.argmax(gain)
        #print(selected_attr)
        
        if np.all(gain < 1e-6):
            return stats.mode(y)[0]


        sets = self.attSplit(x[:, selected_attr])
        
        
        res = {}
        for k, v in sets.items():
            y_subset = y.take(v, axis=0)
            x_subset = x.take(v, axis=0)
            
            res["feature=%s=%d" % (selected_attr, k)] = self.train(x_subset, y_subset,depth, currentDepth+1)
            
            
        if currentDepth == depth:
            return res
        
             
         
        return res

    def singlePred(self, x, model):
       
        for i in model.keys():
            cond = str(i).split("=")
            
            if x[int(cond[1])] == int(cond[2]):
                #print(cond)
                try:        
                    return self.singlePred(x, model[i])
                except:
                    
                    return int(model[i])
                    
    def predict(self, x, model):
        predictions = np.empty(len(x))
        for i in range(0,len(x)):
            predictions[i] = self.singlePred(x[i], model)    
        
        return predictions
            
            
    

if __name__ == '__main__':
    a = DecisionTreeHW()
    def extractFeatures(fileName: str) -> dict: 
        f = open(fileName, 'r')
        labelList = []
        firstName = []
        middleName = []
        lastName = []
        longFirst = []
        middle = []
        sameLetter = []
        firstBeforeLast = []
        vowelSecond = []
        evenLast = []
        vowels = ['a','e','i','o','u']


        while True:
            line = f.readline()
            lineData = str.split(line, ' ')
            if not line or len(lineData) == 1: 
                break
            else:            

                if lineData[0] == '+':
                    labelList.append(1)
                else:
                    labelList.append(0)
                firstName.append(lineData[1])
                lastName.append(lineData[len(lineData)-1])
                try:
                    if str(lineData[1][1]).lower() in vowels:
                        vowelSecond.append(1)
                    else:
                        vowelSecond.append(0)
                except:
                    vowelSecond.append(0)
                
                if len(lineData[len(lineData)-1])%2 == 0:
                    evenLast.append(1)
                else:
                    evenLast.append(0)

                if lineData[1][0] < lineData[len(lineData)-1][0]:
                    firstBeforeLast.append(1)
                else:
                    firstBeforeLast.append(0)
                    
                if lineData[1][0].lower() == lineData[1][len(lineData[1])-1].lower():
                    sameLetter.append(1)
                else:
                    sameLetter.append(0)
                if len(lineData[1]) > len(lineData[len(lineData)-1]):
                    longFirst.append(1)
                else:
                    longFirst.append(0)
                
                if len(lineData) == 4:
                    middleName.append(lineData[3])
                    middle.append(1)
                else:
                    middleName.append('')
                    middle.append(0)
        return [np.array([longFirst, middle, sameLetter, firstBeforeLast, vowelSecond, evenLast]).T, np.array(labelList)]

    data = extractFeatures('updated_train.txt')

    finalX = data[0]
    
    labelList = data[1]
    
    model = a.train(finalX, labelList,4)
    data2 = extractFeatures('updated_test.txt')
    
    testX = data2[0]
    testLabels = data2[1]

    pprint(model)
    pred = a.predict(finalX, model)
    count = 0
    for i in range(0,len(pred)):
        if pred[i] == labelList[i]:
            count+=1
    print('Error Rate on Training = ' + str((1 - (count/len(pred)))*100) + '%')

    pred = a.predict(testX, model)
    count = 0
    for i in range(0,len(pred)):
        if pred[i] == testLabels[i]:
            count+=1
    
    print('Error Rate on Test = ' + str((1 - (count/len(pred)))*100) + '%')
    data0 = extractFeatures('updated_training00.txt')
    data1 = extractFeatures('updated_training01.txt')
    data2 = extractFeatures('updated_training02.txt')
    data3 = extractFeatures('updated_training03.txt')
    
    l=[1,2,3,4,5,10,15,20]
    #4-fold cross validation implementation
    for depth in l:
        

        trainX = np.concatenate((data1[0], data2[0], data3[0]))
    
        trainLabels = np.concatenate((data1[1], data2[1], data3[1]))
        model2 = a.train(trainX, trainLabels, depth)
        testLabels = data0[1]
    
        pred = a.predict(data0[0], model2)
        count = 0
        for i in range(0,len(pred)):
            if pred[i] == testLabels[i]:
                count+=1
        print('Cross Validation Error Depth ' + str(depth) + ' = ' + str((1 - (count/len(pred)))*100) + '%')
    for depth in l:
        

        trainX = np.concatenate((data0[0], data1[0], data2[0]))
    
        trainLabels = np.concatenate((data0[1], data1[1], data2[1]))
        model2 = a.train(trainX, trainLabels, depth)
        testLabels = data3[1]
    
        pred = a.predict(data3[0], model2)
        count = 0
        for i in range(0,len(pred)):
            if pred[i] == testLabels[i]:
                count+=1
        print('Cross Validation Error Depth ' + str(depth) + ' = ' + str((1 - (count/len(pred)))*100) + '%')
    for depth in l:
        

        trainX = np.concatenate((data0[0], data2[0], data3[0]))
    
        trainLabels = np.concatenate((data0[1], data2[1], data3[1]))
        model2 = a.train(trainX, trainLabels, depth)
        testLabels = data1[1]
    
        pred = a.predict(data1[0], model2)
        count = 0
        for i in range(0,len(pred)):
            if pred[i] == testLabels[i]:
                count+=1
        print('Cross Validation Error Depth ' + str(depth) + ' = ' + str((1 - (count/len(pred)))*100) + '%')
    for depth in l:
        

        trainX = np.concatenate((data0[0], data1[0], data3[0]))
    
        trainLabels = np.concatenate((data0[1], data1[1], data3[1]))
        model2 = a.train(trainX, trainLabels, depth)
        testLabels = data2[1]
    
        pred = a.predict(data2[0], model2)
        count = 0
        for i in range(0,len(pred)):
            if pred[i] == testLabels[i]:
                count+=1
        print('Cross Validation Error Depth ' + str(depth) + ' = ' + str((1 - (count/len(pred)))*100) + '%')

    
    

       
    


    
   
    

    

