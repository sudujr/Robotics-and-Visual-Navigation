# Import Necessary Packages
import numpy as np 
from matplotlib import pyplot as plt 
from statistics import mean, stdev
import pandas as pd
import copy

class LinearRegression:

    def load_file(self, fileName):
        df = pd.read_csv(fileName, header = None)
        df.columns = ['X', 'Y']
        return df.drop(['Y'], axis = 1) , df.drop(['X'], axis = 1)

    def add_bias(self, X):
        return pd.Series(1, index=X.index, name = 'Bias')
    
    def add_poly_features(self, bias, X):
        phi = pd.concat([bias,X], axis = 1)
        phi['X1'] = X['X'] ** 1
        phi['X2'] = X['X'] ** 2
        phi['X3'] = X['X'] ** 3
        phi = phi.drop(['X'], axis = 1)
        return phi

    def initilialize_theta(self, n):
        theta0 = np.array([0.0] * n)
        return theta0

    def hypothesis(self, X, theta):
        y_pred = theta * X 
        return np.sum(y_pred, axis = 1)
    def gradient_descent(self, X, Y, m, n, theta, alpha = int(0.01), max_iter = int(1000)):
        for i in range(max_iter):
            print('Iteration : ', i)
            y_pred = self.hypothesis(X, theta)
            for j in range(n):
               print(y_pred - Y)
        return theta



        

def main():
    lr = LinearRegression()
    X, Y= lr.load_file('data_regress.txt')
    bias = lr.add_bias(X)
    phi = lr.add_poly_features(bias, X)
    m,n = phi.shape
    theta0 = lr.initilialize_theta(n)
    theta = lr.gradient_descent(phi, Y, m , n, theta0)
    y_hat = theta * phi
    plt.plot(X,y_hat, 'bx')
    plt.show()


    

if __name__ == "__main__":
    main()
