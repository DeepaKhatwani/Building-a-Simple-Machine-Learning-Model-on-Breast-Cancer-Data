# TRAIN A BREAST CANCER CLASSIFICATION ALGORITHM

import sys
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from sklearn import datasets #library to load dataset
import sklearn

# ################### TO KNOW BASIC INFORMATION OF DATASET ########################################
'''
Binary Classification dataset
Target Names : 'malignant', 'benign'
'''
#Method 1 to know the target
#k, y = sklearn.datasets.load_wine(return_X_y=True)
# print(y)

# ################### TO LOAD CANCER DATASET ######################################################
raw_data = datasets.load_breast_cancer()
print(raw_data)
print(type(raw_data))
print(raw_data.feature_names)
print(raw_data.target_names) 
df_feature = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
df = df_feature
df['target'] = raw_data['target']
df.head()

# ################## TO ANALYSE DATA USING FUNCTIONS ###################################
df.shape #31 columns
df.info() #no null data
df.describe().T
df.groupby('target').describe().T

X = df.iloc[:, 1:31].values # Feature data
Y = df.iloc[:, 30].values # Target data

print(X)
print(Y)

# TO CHECK MISSING DATA OR NULL VALUES
df.isnull().sum()
df.isna().sum()
# No missing or null data



# ################## DATA VISUALIZATION ###################################
# to display histogram
for col_name in df.columns:
    print(col_name)
    statement = "df.hist(column = '" + col_name +"', by = 'target');"
    exec(statement)    
    plt.show()

# ################## TO FIND RELATIONS BETWEEN DATA USING CORRELATION MATRIX ######################
corr_matrix = df.corr()
print(corr_matrix.T)
print(corr_matrix["target"].sort_values(ascending=False))


#colum_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
# Correlation matrix
correlations = df.corr()
# Plot figsize
fig, ax = plt.subplots(figsize=(10, 10))
# Generate Color Map
colormap = sns.diverging_palette(220, 10, as_cmap=True)
# Generate Heat Map, allow annotations and place floats in map
sns.heatmap(correlations, cmap=colormap, annot=True, fmt=".2f")
ax.set_xticklabels(
    df.columns,
    rotation=45,
    horizontalalignment='right'
);
ax.set_yticklabels(df.columns);
plt.show()

# To remove unwanted columns #############################################################

df.columns
data_update = df
data_update.drop(['smoothness error','mean fractal dimension','texture error','symmetry error', 'fractal dimension error'], axis=1, inplace=True)
data_update.columns
# ################## TO SPLIT DATA INTO TEST AND TRAIN ############################################
data = df.values
type(data) #always use this format

X = data[:,:-1]
X

y = data[:,-1]
y=y.astype('int')
y

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)
X_train.shape
X_test.shape
y_train.shape
y_test.shape

# ################## TO APPLY MACHINE LEARNING ALGORITHM ##########################################

#------------------- Machine learning 1 ------------
print('---------- LogisticRegression--------------------------')
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine species for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#------------------- Machine learning 2 ------------
print('---------- KNeighborsClassifier--------------------------')
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 3 ------------
print('---------- DecisionTreeClassifier--------------------------')

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 4 ------------
print('---------- SVC--------------------------')

from sklearn.svm import SVC

classifier = SVC()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 5 ------------
print('---------- GaussianNB--------------------------')

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 6 ------------
print('---------- RandomForestClassifier--------------------------')

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 7 ------------
print('---------- MLPClassifier--------------------------')

from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# RESULT -  RandomForestClassifier is the best classifier - 97.37

# 

