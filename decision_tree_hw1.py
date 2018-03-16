import typing
import math
import operator
import numpy as np
from pprint import pprint

class DecisionTreeHW(object):

    def __init__(self) -> None:
        print('ID3 under way...')

    def split(self, data: list) -> dict:
        return {val: (data==val).nonzero()[0] for val in np.unique(data)}

    def entropy(self, data: list) -> float:
        res = 0
        val, counts = np.unique(data, return_counts=True)
        freqs = counts.astype('float')/len(data)
        for p in freqs:
            if p != 0.0:
                res -= p * np.log2(p)
        return res
        
    def informationGain(self, y: list, x: list) -> float:

        res = self.entropy(y)

        # We partition x, according to attribute values x_i
        val, counts = np.unique(x, return_counts=True)
        freqs = counts.astype('float')/len(x)

        # We calculate a weighted average of the entropy
        for p, v in zip(freqs, val):
            res -= p * self.entropy(y[x == v])

        return res

    def uniform(self, data: list):
        return len(set(data)) == 1

    def recursion(self, x: list, y: list):
        
        if self.uniform(y) or len(y) == 0:
            return y
        
        gain = np.array([self.informationGain(y, x_attr) for x_attr in x.T])
        
        selected_attr = np.argmax(gain)
        if selected_attr == 0:
            attrName = 'LongFirst'
        elif selected_attr == 1:
            attrName = 'middle'
        elif selected_attr == 2:
            attrName = 'sameLetter'
        elif selected_attr == 3:
            attrName = 'firstBeforeLast'
        elif selected_attr == 4:
            attrName = 'vowelSecond'
        else:
            attrName = 'evenLast'

        if np.all(gain < 1e-6):
            return y


        sets = self.split(x[:, selected_attr])
        
        res = {}
        for k, v in sets.items():
            y_subset = y.take(v, axis=0)
            x_subset = x.take(v, axis=0)

            res["%s = %d" % (attrName, k)] = self.recursion(x_subset, y_subset)

        return res

    

if __name__ == '__main__':
    a = DecisionTreeHW()
    f = open('training.data', 'r')
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
    

    finalX = np.array([longFirst, middle, sameLetter, firstBeforeLast, vowelSecond, evenLast]).T
    labelList = np.array(labelList)
    pprint(a.recursion(finalX, labelList))

    
   
    

    

