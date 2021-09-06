#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                names= ["SepalLength","SepalWidth","PetalLength","PetalWidth","Species"])
#Spliting the data into: training- 75% and testing- 25%

X=df.drop(columns=['Species'])
Y = df['Species']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25)

k = 10

modelkNN = KNeighborsClassifier(k)
modelkNN.fit(x_train,y_train)
accuracy = modelkNN.score(x_test,y_test)

print(f"k = {k}, accuracy = {accuracy}")



from sklearn.model_selection import cross_val_score

k_scores = []
for k in range(1,41):
    model = KNeighborsClassifier(k)
    scores = cross_val_score(model, X,Y,cv=20,scoring='accuracy')
    k_scores.append(scores.mean())

plt.plot(range(1,41),k_scores,'-')
plt.plot(range(1,41),k_scores,'bo')
plt.xlabel("k-values")
plt.ylabel("accuracy")
plt.title("kNN with 20-fold CV")
plt.savefig("accuracy_kNN.png")
plt.show()


best_k = k_scores.index(max(k_scores))
print(f'best_k = {best_k}, accuracy = {accuracy}')
