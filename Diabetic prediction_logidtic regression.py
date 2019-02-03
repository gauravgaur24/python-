import pandas as pd
import sklearn
import numpy as np
from sklearn import metrics

from sklearn import model_selection 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

col= ['Pregnancies' ,'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
df = pd.read_csv("diabetes.csv", header=1, names=col)

#splitting of data required
feature_cols = ['Pregnancies' ,'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
x=df[feature_cols]
y=df['Outcome']
print(x,y)

X_train,X_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.25,random_state=0)

print(X_train,X_test,y_train,y_test)

logreg=LogisticRegression()
print(logreg.fit(X_train,y_train))

y_pred=logreg.predict(X_test)
print(y_pred)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
