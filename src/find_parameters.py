from numpy.__config__ import show
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


dataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                names= ["SepalLength","SepalWidth","PetalLength","PetalWidth","Species"])

##Split data
"""
train = dataframe.iloc[:35].append(dataframe.iloc[50:85]).append(dataframe.iloc[100:135])
test = dataframe.iloc[35:50].append(dataframe.iloc[85:100]).append(dataframe.iloc[135:])

X_train, y_train = train.iloc[:,:-1], train.iloc[:,-1]
X_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]
"""

X_train, X_test, y_train, y_test = train_test_split(dataframe.iloc[:,:-1], dataframe.iloc[:,-1])

##Data generator
from sklearn.datasets import make_classification

X_g, y_g = make_classification(n_samples=1000, n_features=4,n_classes=3,n_informative=4,n_redundant=0,n_repeated=0, random_state=0)

X_g_train, X_g_test, y_g_train, y_g_test = train_test_split(X_g, y_g)

def SVC_choose_gamma(X, y, show_results=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    gamma_values = [0.01, 0.1, 1, 10]
    accuracy_scores = []

    for g in gamma_values:
        model = svm.SVC(gamma=g).fit(X_train, y_train)
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_pred, y_test)
        accuracy_scores.append(accuracy)
        if(show_results):
            print("gamma: %2f, accuracy score: %2f" %(g, accuracy))

    ind = np.argmax(accuracy_scores)
    return gamma_values[ind]

def SVC_choose_C(X, y, show_results=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    c_values = [0.1, 1, 10, 100]
    accuracy_scores = []

    for c in c_values:
        model = svm.SVC(C=c).fit(X_train, y_train)
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_pred, y_test)
        accuracy_scores.append(accuracy)
        if(show_results):
            print("C: %2f, accuracy score: %2f" %(c, accuracy))

    ind = np.argmax(accuracy_scores)
    return c_values[ind]


final_model = svm.SVC(gamma = SVC_choose_gamma(X_g, y_g,show_results=True), C = SVC_choose_C(X_g, y_g, show_results=True))
fit = final_model.fit(X_train, y_train)

y_pred = fit.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
print("Train/test accuracy: %2f" %accuracy)


##Bootstrap
from sklearn.utils import resample

bs = resample(dataframe, n_samples = 1000,replace = True)

X_bs_train, X_bs_test, y_bs_train, y_bs_test = train_test_split(bs.iloc[:,:-1], bs.iloc[:,-1])

m = svm.SVC(gamma = SVC_choose_gamma(X_g, y_g), C = SVC_choose_C(X_g, y_g))
bs_fit = m.fit(X_bs_train, y_bs_train)

y_bs_pred = bs_fit.predict(X_bs_test)
bs_accuracy = accuracy_score(y_bs_pred, y_bs_test)
#print("Bootstapped accuracy: %2f" %bs_accuracy)


