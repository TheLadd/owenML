# Using my own implementation of linear regression via gradient descent, I train a model
# against a training set to determine an appropriate bias and set of weights in order to 
# achieve an accurate estimate of y
#
# This application is done in the context of UCI's Boston Housing dataset 

# Related third party imports.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 

# Local library imports
import owenML



# 1. Import and categorize data
names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
    'AGE',  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE'
]
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                 header=None,delim_whitespace=True,names=names,na_values='?')
X = df[['CRIM', 'RM', 'LSTAT']]
y = df['PRICE']

# 2. Pre-processing stage
#   a. Add column of 1's to X, in order to calculate b through dot product 
X = X.assign(ones=1)
X = X[['ones', 'CRIM', 'RM', 'LSTAT']]
X = X.values
y = y.values

#   b. Split data into training and test (or validation) sets    
(Xtr, Xts, ytr, yts) = train_test_split(X, y, test_size=0.3)

# 4. Run gradient descent at 2 different step sizes
(w, all_cost,iters) = owenML.linear_regression_gd(Xtr,ytr,learning_rate = 0.00001,max_iter = 1000, tol=pow(10,-6))  
plt.figure(0)
plt.semilogy(all_cost[0:iters])    
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Training loss')  

(w, all_cost,iters) = owenML.linear_regression_gd(Xtr,ytr,learning_rate = 0.000001,max_iter = 1000, tol=pow(10,-6))  
plt.semilogy(all_cost[0:iters])    

plt.legend(['.00001', '.000001'])
plt.savefig("./figs/housing_weight_descent.png")
plt.clf()

# Here is where I'd plot the real points against my regression line Xw, if the 
# data points weren't 5-dimensional :D


