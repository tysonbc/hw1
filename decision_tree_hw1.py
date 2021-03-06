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

    def setValue(self, y_pred: dict, ind: int, val: int) -> dict:
        y_pred[ind] = val
        return y_pred
    def predict(self, model: dict, x: dict, w: bool=True, y_shape: int=None) -> dict:
        

        try:
            predictions = self.predictions
        except:
            if not y_shape:
                predictions = np.zeros(len(x))
            else:
                predictions = np.zeros(y_shape)
        

        for i in model.keys():
            #print(i)
            cond = str(i).split("=")
            
            sets = self.attSplit(x[:, int(cond[1])])
            v = sets[int(cond[2])]
            x_subset = x.take(v, axis=0)
            #np.unique(x_subset, axis=0))
                #break
            #print(cond)
            #print(model)
            try:
                if y_shape:
                    self.predict(model[i],x_subset,False, y_shape = y_shape)
                else:
                    self.predict(model[i],x_subset,False, y_shape = len(x))
            except:
                
                for p in np.unique(x_subset, axis = 0):
                    #print(p)
                    for y in range(0,len(x)-1):
                        #print(x[y])
                        if str(p) == str(x[y]):
                            
                            #print(model[i])
                            try:
                                self.predictions = self.setValue(predictions, y, int(model[i]))
                                #print(len(self.predictions))
                            except:
                                pass
                
                            
                            #predictions[y] = float(model[i])
                            #print(predictions[y])
                           
                            
                            #print(int(model[i]))
                            #predictions = np.put(predictions,y, int(model[i]))
                
                    #print(model[i])
            
        return self.predictions
        
            
            
    

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
    
    model = a.train(finalX, labelList,2)
    pprint(model)
    
   
    


    
   
    

    

