# Using my own implementation of the K-nearest neighbors classifier, this program
# uses randomly determined subsets of seaborns penguin dataset as a "training" set 
# and a test set. 
# 
# It performs 100 assesments of randomized test/training data for all K in the 
# range [1, 10]. It then calculates and plots the average accuracy -- as well as the standard 
# deviation -- for all values of K.
#
# Drawn figures can be found in /figs

# Related third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Local library imports
import owenML

# Load penguin data
data = sns.load_dataset("penguins") # load penguins dataset from seaborn
data = data.dropna() # drop samples with missing values (NaN) 

# Show diffenreces amongst the features of different penguin species
fig = sns.pairplot(data, hue="species")
plt.savefig("./figs/penguin_pairplot.png")
plt.clf()

# Convert data into np.ndarray's for our inputs, X, and our labels, all_labels
X = data[['bill_length_mm','bill_depth_mm','flipper_length_mm', 'body_mass_g']].values
all_labels = data['species'].values
unique_labels = np.unique(all_labels)

# 1. Preprocessing Stage
  # a. Normalize our data for more accurate predictions
xNorm = owenML.normalize(X)


# 2. Processing Stage
  # a. Perform 100 assesments of randomized test data in the context of randomized training data
  #    for all values of K in [1, 10]    
K = 10
kAccuracy = []
errCnt = {'Adelie':0, 'Chinstrap':0, 'Gentoo':0} # To keep track of how many of each species we misidentify
for i in range(K):
  kAccuracy.append([])
  for j in range(100):
    (training_data, test_data, training_labels, test_labels) = train_test_split(xNorm, all_labels, test_size=0.3)
    pred_labels = owenML.knnclassify(test_data, training_data, training_labels, i+1)
    kAccuracy[i].append(sum(test_labels == pred_labels)/len(test_labels))
    errors = np.array(pred_labels)[test_labels != pred_labels]
    for x in errors:
      errCnt[x] += 1

  # b. Calculate average accuracy and standard error of knnclassify() for all iterations
kMean = []
kStd = []
for i in range(10): 
  kMean.append(np.mean(kAccuracy[i]))
  kStd.append(np.std(kAccuracy[i]))


# 3. Plot
print(errCnt)
plt.errorbar(x=range(1, K+1), y=kMean, yerr=kStd)
plt.savefig("./figs/knn_penguins.png")