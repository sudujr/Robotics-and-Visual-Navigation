import numpy as np 
import pandas as pd 

df = pd.read_csv('data_regress.txt', header = None)
df.columns = ['X1', 'Y']
phi = df.drop(['Y'], axis = 1)
Y = df.drop(['X1'], axis = 1)
phi = pd.concat([pd.Series(1, index = df.index, name='Bias'), phi], axis = 1)
phi['X2'] = phi['X1'] ** 2
phi['X3'] = phi['X1'] ** 3
print(phi.head)
print(phi)