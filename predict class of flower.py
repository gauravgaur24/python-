from __future__ import print_function

import imp
import pandas as pd  #data frame
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt #plotting of graph 

from sklearn import model_selection   #selecte the logic/algo
from sklearn.metrics import classification_report # grouping/ categorization
from sklearn.metrics import confusion_matrix#fold(no of test) test 
from sklearn.metrics import accuracy_score #
from sklearn.linear_model import LogisticRegression #


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
cols=['sepal-length','sepal width','petal-length','petal-width','class']
data=pd.read_csv(url,names=cols)  #load data from url 

#print(data)
#print(data.describe())

#we can plot histogram
#data.plot(kind='box',subplots=True,layout=(2,2),sharex=False, sharey=False)
#data.hist()
#plt.show()

#scatter plot matrix
#scatter_matrix(data)
#plt.show()

flower_data=data.values# converting the data into list 
#print(flower_data)

# the data consist of numeric and alphabetic class
# with the help of slicer taken only data of numeric terms

x=flower_data[:,0:4]
y=flower_data[:,4]

#print(x)
#print(y)

v=0.30# validation size .
s=5 # seed
scoring='accuracy'
xtrain,xtest,ytrain,ytest= model_selection.train_test_split(x,y,test_size=v, random_state=s)

print(xtest)# test data 
print(ytest)# test data 

# check all algo and append the result in the list
models=[]
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluvate each model
result=[]
names=[]

for name,m in models:
    kfold=model_selection.KFold(n_splits=10, random_state=s)
    cv_results = model_selection.cross_val_score(m, xtrain, ytrain, cv=kfold, scoring=scoring)
    result.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#plotting the comparison of different models on the graph 
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(result)
ax.set_xticklabels(names)
plt.show()

knn = KNeighborsClassifier()
knn.fit(xtrain, ytrain)
predictions = knn.predict(xtest)
print(accuracy_score(ytest, predictions))
print(confusion_matrix(ytest, predictions))
print(classification_report(ytest, predictions))

print("svc model")
s=SVC()
s.fit(xtrain,ytrain)
predictions = s.predict(xtest)
print(accuracy_score(ytest, predictions))
print(confusion_matrix(ytest, predictions))
print(classification_report(ytest, predictions))


print("prediction-",predictions)









