import typing
import numpy as np

class DecisionTreeHW(object):

    def __init__(self) -> None:
        print('ID3 under way...')

    def split(self, feature: list) -> dict:
        return {value: (feature==value).nonzero()[0] for value in np.unique(feature)}

    def entropy(self, feature: list) -> float:
        res = 0
        val, counts = np.unique(feature, return_counts=True)
        freqs = counts.astype('float')/len(feature)
        for p in freqs:
            if p != 0.0:
                res -= p * np.log2(p)
        return res

if __name__ == '__main__':
    a = DecisionTreeHW()
    f = open('training.data', 'r')
    label = []
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
            

            label.append(lineData[0])
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
    
    b = a.entropy(feature = longFirst)
    print(b)

    

