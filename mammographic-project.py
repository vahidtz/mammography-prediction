# -*- coding: utf-8 -*-
# Code by: Vahid Mohammadnia--
import numpy as np
import pandas as pd
Raw_Data = pd.read_csv("D:/25a-data-science/mammographic_masses.data.txt")
Raw_Data=Raw_Data.replace("?",np.nan)
Column_Names = ['BI-RADS', 'Age','Shape','Margin','Density', 'Severity']
Raw_Data.columns = Column_Names
print (Raw_Data.describe(include = 'all'))
Raw_Data = Raw_Data.dropna()
all_features = Raw_Data[Column_Names].drop(['BI-RADS', 'Severity'], axis=1).values
all_classes = Raw_Data['Severity'].values
labels = Column_Names[1:len(Column_Names)-1]
#.............................................Scaling the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(all_features)
all_features2 = scaler.transform(all_features)
#......................................................Splitting the data int traing and test data = 25%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(all_features2, all_classes, test_size=0.25)
# .....................................................Applying Decision Tree

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train ,y_train)

from IPython.display import Image  
from sklearn.externals.six import StringIO  
import pydotplus

dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=labels)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png()) 

# ................................................Measuring the test data error:
result = clf.score(X_test, y_test)
print("Accuracy on test data by Desicion Tree is:",result)
# ....................................................K-Fold_desicion Tree
from sklearn.model_selection import cross_val_score
clf2 = tree.DecisionTreeClassifier()
cv_scores2 = cross_val_score(clf2, all_features2, all_classes, cv = 10)
print("cv_Score2 by Desicion Tree is:", cv_scores2.mean())
cv_scores = cross_val_score(clf, all_features2, all_classes, cv = 10)
print("cv_Score by Desicion Tree is:", cv_scores.mean())

#.....................................................................................................Random Forest 
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(X_train, y_train)
# ......................................................................test data error by Random Forest 
result = clf.score(X_test, y_test)
print("Accuracy on test data by Random Forest is:",result)
# ................................................. .....................K-Fold_Random Forest 
clf = RandomForestClassifier(n_estimators=100)
cv_scores = cross_val_score(clf, all_features2, all_classes, cv = 10)
print("cv_Score by Random Forest is:", cv_scores.mean())
#....................................................................................................SVM with a linear kernel
from sklearn import svm, datasets
C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
# ......................................................................test data error by SVM 
result = svc.score(X_test, y_test)
print("Accuracy on test data by SVM is:",result)
# ................................................. .....................K-Fold by SVM
model = svm.SVC(kernel='linear', C=C)
cv_scores = cross_val_score(model, all_features2, all_classes, cv = 10)
print("cv_Score by SVM is:", cv_scores.mean())
#....................................................................................................KNN with a linear kernel
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train) 
#......................................................................Test data error by KNeighborsClassifier 
result = neigh.score(X_test, y_test)
print("Accuracy on test data by KNeighborsClassifier is:",result)
# ................................................. .....................K-Fold by SVM
model = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(model, all_features2, all_classes, cv = 10)
print("cv_Score by KNeighborsClassifier is:", cv_scores.mean())
#..................................         .Choosing different number of neigbors
for k in range(48,50):
    model = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(model, all_features2, all_classes, cv = 10)
    print("cv_Score by KNeighborsClassifier for n_neighbors =%d is:" %k, cv_scores.mean())
    
#................................................................................................Naive Bayes     
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

classifier = MultinomialNB()
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train2 = scaler.transform(X_train)
classifier.fit(X_train2, y_train)
#...............................test data error by Naive Bayes
result = classifier.score(scaler.transform(X_test), y_test)
print("Accuracy on test data by Naive Bayes is:",result)

# ................................................. .....................K-Fold by Naive Bayes
all_features2 = scaler.transform(all_features)
cv_scores = cross_val_score(classifier, all_features2, all_classes, cv = 10)
print("cv_Score by Naive Bayes is:", cv_scores.mean())

#....................................................................................................SVM with a rbf kernal
C = 1.0
svc = svm.SVC(kernel='rbf', C=C).fit(X_train, y_train)
# ......................................................................test data error by SVM 
result = svc.score(X_test, y_test)
print("Accuracy on test data by SVM with rbf kernals is:",result)
# ................................................. .....................K-Fold by SVM
model = svm.SVC(kernel='rbf', C=C)
cv_scores = cross_val_score(model, all_features2, all_classes, cv = 10)
print("cv_Score by SVM with rbf kernals is:", cv_scores.mean())


#....................................................................................................SVM with a sigmoid kernal
C = 1.0
svc = svm.SVC(kernel='sigmoid', C=C).fit(X_train, y_train)
# ......................................................................test data error by SVM 
result = svc.score(X_test, y_test)
print("Accuracy on test data by SVM with sigmoid kernals is:",result)
# ................................................. .....................K-Fold by SVM
model = svm.SVC(kernel='sigmoid', C=C)
cv_scores = cross_val_score(model, all_features2, all_classes, cv = 10)
print("cv_Score by SVM with sigmoid kernals is:", cv_scores.mean())

#....................................................................................................Logistic Regresisons   
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
model = LR.fit(X_train, y_train)
# ............................................................test data error by Logistc REgresison 
result = model.score(X_test, y_test)
print("Accuracy on test data by Logistic Regression with sigmoid kernals is:",result)
# ................................................. .....................Logistic Regression by Logistc Regresison 

cv_scores = cross_val_score(LR, all_features2, all_classes, cv = 10)
print("cv_Score by Logistic Regression with sigmoid kernals is:", cv_scores.mean())

#.............................................................Keras Neural Network
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation
# def create_model():
model = Sequential()
model.add(Dense(8, input_dim=4, kernel_initializer='normal',activation='relu'))
model.add(Dense(4, kernel_initializer= 'normal',activation='relu'))
model.add(Dense(1,activation ='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])
    # return model
history = model.fit(X_train, y_train,
                    batch_size=100,
                    epochs=30,
                    verbose=2, validation_data=(X_test, y_test)
                    )
# ............................................................test data error by Keras

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
