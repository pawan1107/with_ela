import os
import pandas as pd 

path = os.getcwd() + "/dataset/ela/"

data = []
dir = ['fake', 'real']
for d in dir:
    for f in os.listdir(path+d):
        a = path+d+"/"+f
        if d == 'real':
            b = 0
        elif d == 'fake':
            b = 1
        data.append([a,b])

df = pd.DataFrame(data)
df.to_csv('dataset.csv')
