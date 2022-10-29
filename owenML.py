# My implementation of some common (and maybe some not-so-common) algorithms/functions
# to the field of ML

import numpy as np
from math import exp

# -------------- KNN --------------
def distance(x,y,p=2):
    """calculates Lp distance between point x and point y
    Args:
        x (np.ndarray): datapoint x
        y (np.ndarray): datapoint y
        p (int): order of Lp norm
    """
    if len(x) != len(y):
      print("distance(x, y, p) was given two vectors (x, y) whose number of dimensions didn't match!")

    pDiffSum = 0
    for i in range(len(x)):
      diff = abs(x[i] - y[i])
      pDiffSum += diff ** p
    
    return pDiffSum ** (1/p)

def normalize(X):
    """Returns a version of X where all elements in X[i] 
    (for i in len(X)) have been mapped from [min(X[i]), max(X[i])] to [0, 1]
    Args:
        X: np.ndarray of shape (i, j), where all elements are floats or ints
    """
    Xt = X.transpose()
    rowRanges = [(min(Xt[i]), max(Xt[i])) for i in range(len(Xt))]

    xNorm = X.copy()
    for i in range(len(xNorm)):
        for j in range(len(xNorm[i])):
            xNorm[i][j] = np.interp(X[i][j], [rowRanges[j][0], rowRanges[j][1]], [0, 1])

    return xNorm

def knnclassify(test_data, training_data, training_labels, K=1):
  """KNN classifier
    Args:
      test_data (numpy.ndarray): Test data points.
      training_data (numpy.ndarray): Training data points.
      training_labels (numpy.ndarray): Training labels.
      K (int): The number of neighbors.
    
    Returns:
      pred_labels: contains the predicted label for each test data point, have the same number of rows as 'test_data'
  """
  pred_labels = []

  for i in range(len(test_data)):
    # 1. Compute distance from test_data[i] to all points in training_data
    dists = []
    for j in range(len(training_data)):
      dists.append(distance(test_data[i], training_data[j]))

    # 2. Find the indices for the k points in training_data with smallest distances from test_data[i], store in kIndices[]
    kIndices = sorted(range(len(dists)), key = lambda sub: dists[sub])[:K]
      
    # 3. Assign pred_labels[i] the mode of the labels of training_data[kIndices]
    kLabels = [training_labels[j] for j in kIndices]
    kLabelsUnq = list(np.unique(kLabels))
    pred_labels.append(max(kLabelsUnq, key=kLabels.count))

  return pred_labels

# ----------------- Linear Regression --------------------
def compute_cost(X,w,y):
    """
    The squared loss function, ||Xw - y||^2
    """
    L = sum([(y[i] - (X[i] @ w))**2 for i in range(len(y))])
    return L

def linGradient(X, w, y):
    """
    Calculates the gradient of the squared loss function, ||Xw - y||^2
    X: np.ndarray of shape (n, d)
    w: np.array of shape (1, d)
    y: np.array of shape (n, 1)
    """
    xw = [i for i in np.array(np.dot(X, w))] # Because dot product returns array of 1d arrays
    return np.dot(X.transpose(), xw-y)
    
def linear_regression_gd(X,y,learning_rate = 0.00001,max_iter=10000,tol=pow(10,-5)):
    """
    Runs a linear regression via gradient descent against the set of points (X, y)

    Returns:
      A set (np.array) of weights, w
    """
    w = np.array([1, 1, 1, 1])
    all_cost = []
    it = 0
    
    for i in range(max_iter):
        all_cost.append(compute_cost(X,w,y))
        if it > 0 and np.absolute(all_cost[it]-all_cost[it-1])/all_cost[it-1] <= tol:
            break
        w = w - learning_rate * linGradient(X, w, y)
        it += 1

    return w, all_cost, it


# ------------------ Logistic Regression -------------------
def predict(X,w):
  """
  X: np.ndarray of shape (n, d)
  w: np.array of shape (1, d)

  Returns: 
    Hard label of Xw, yhat (i.e., yhat[i] = 1 if Xw[i] > 0, else yhat[i] =  0)
  """
  yhat = [np.dot(w, X[i]) > 0 for i in range(len(X))]
  return yhat

def sigmoid(z):
    """
    z = some scalar
    """
    return 1/(1+exp(-z))

def log_cost(X,w,y):
    """
    X: shape = N*d
    w: shape = d
    y: shape = N
    """
    pr1 = [y[i]*np.log(sigmoid(np.dot(w.transpose(), X[i]))) for i in range(len(y))]
    pr0 = [(1-y[i])*np.log(1 - sigmoid(np.dot(w.transpose(), X[i]))) for i in range(len(y))]
    return -1*sum(pr1 + pr0)

def logGradient(X, w, y):
    """
    Calculates the gradient of logistic cost as a function of X, w, and y
    X: np.ndarray of shape (n, d)
    w: np.array of shape (1, d)
    y: np.array of shape (n, 1) 
    """
    seq = [(sigmoid(np.dot(w.transpose(), X[i])) - y[i])*X[i] for i in range(len(y))]
    return sum(seq)

def logistic_regression_gd(X,y,learning_rate = 0.00001,max_iter=1000,tol=pow(10,-5)):
    """
    Runs a logistic regression via gradient descent for given points (X, y)

    Returns:
      A set (np.array) of weights, w
    """
    w = np.array([1 for i in range(len(X[0]))]) # d 1's
    all_cost = []
    it = 0

    for i in range(max_iter):
        all_cost.append(log_cost(X, w, y))
        if it > 0 and np.absolute(all_cost[it]-all_cost[it-1])/all_cost[it-1] <= tol:
            break
        w = w - learning_rate * logGradient(X, w, y)
        it += 1

    return w, all_cost, it