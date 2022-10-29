# Project desc.


# Related third party imports.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 

# Local library imports
import owenML

# Load seaborn breast cancer dataset
names = ['id','thick','size_unif','shape_unif','marg','cell_size','bare',
         'chrom','normal','mit','class']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/' +
                 'breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                names=names,na_values='?',header=None)

# Be picky about our data
xnames =['size_unif','marg'] 
X = df[xnames].assign(ones=1)
X = X[['ones', 'size_unif', 'marg']]
X = np.array(X)

yraw = np.array(df['class'])
BEN_VAL = 2   # value in the 'class' label for benign samples
MAL_VAL = 4   # value in the 'class' label for malignant samples
y = (yraw == MAL_VAL).astype(int)

# Split into training and validation sets
Xtr, Xts, ytr, yts = train_test_split(X,y, test_size=0.30)

# Run two "sessions" of logisitic gradient descent against our data, with different step sizes
(w, all_cost,iters) = owenML.logistic_regression_gd(Xtr,ytr,learning_rate = 0.001,max_iter = 1000, tol=pow(10,-6))  
plt.semilogy(all_cost[0:iters])    
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Training loss') 

yhat = owenML.predict(Xts,w)
acc = np.mean(yhat == yts)
print("Test accuracy = %f" % acc)

(w, all_cost,iters) = owenML.logistic_regression_gd(Xtr,ytr,learning_rate = 0.00001,max_iter = 1000, tol=pow(10,-6))  
plt.semilogy(all_cost[0:iters])    

yhat = owenML.predict(Xts,w)
acc = np.mean(yhat == yts)

plt.legend([".001", ".00001"])
plt.savefig("./figs/brst_cancer_weight_descent.png")

print("Test accuracy = %f" % acc)
