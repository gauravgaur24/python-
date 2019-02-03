import pandas as pd
import sklearn
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection 

import matplotlib.pyplot as plt
import seaborn as sns

col= ['Pregnancies' ,'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
df = pd.read_csv("diabetes.csv", header=1, names=col)

#splitting of data required
feature_cols = ['Pregnancies' ,'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
x=df[feature_cols]
y=df['Outcome']
print(x,y)

X_train,X_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.3,random_state=2)

print(X_train,X_test,y_train,y_test)

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print(y_pred)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
''''
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
print(Image(graph.create_png()))

'''''

























