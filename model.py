# Spyder project settings
.spyderproject
.spyproject

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 18:09:52 2018

@author: ramsey
"""

### Data preprocessing & Cleaning 
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir(r'D:\Machine Learning') 


list(check.columns.values)

WVS = pd.read_excel("WVS.xlsx", na_values='', usecols=[61, 98, 165, 284, 297, 306, 307, 308, 310, 317, 318], 
                           header=0, names=['Economic_Satisfaction', 'Ideology','Religiousity','Vote_AKP', 'Employment', 'Social_Class',
                                            'Income', 'Sex', 'Age', 'Ethnicity', 'Education'])

list(WVS.columns)

WVS['Vote_AKP'].unique()### Dependent Variable: Vote for AKP party 
WVS['Economic_Satisfaction'].unique()### Dependent Variable: Vote for AKP party 
WVS['Ideology'].unique()### Dependent Variable: Vote for AKP party 
WVS['Religiousity'].unique()### Dependent Variable: Vote for AKP party 
WVS['Employment'].unique()### Dependent Variable: Vote for AKP party 
WVS['Social_Class'].unique()### Dependent Variable: Vote for AKP party 
WVS['Income'].unique()### Dependent Variable: Vote for AKP party 
WVS['Sex'].unique()### Dependent Variable: Vote for AKP party 
WVS['Age'].unique()### Dependent Variable: Vote for AKP party 
WVS['Ethnicity'].unique()### Dependent Variable: Vote for AKP party 
WVS['Education'].unique()### Dependent Variable: Vote for AKP party 

len(WVS['Vote_AKP'].value_counts())



from collections import Counter

data = Counter(WVS['Education'])
data.most_common()   # Returns all unique items and their counts
data.most_common(1) 

WVS['Vote_AKP'] = WVS['Vote_AKP'].fillna('TR: AKP - Justice and Development Party') 
WVS['Economic_Satisfaction'] = WVS['Economic_Satisfaction'].fillna('7') 
WVS['Ideology'] = WVS['Ideology'].fillna('6') 
WVS['Religiousity'] = WVS['Religiousity'].fillna('Only on special holy days') 
WVS['Employment'] = WVS['Employment'].fillna('Housewife') 
WVS['Social_Class'] = WVS['Social_Class'].fillna('Lower middle class') 
WVS['Income'] = WVS['Income'].fillna('Fifth step') 
WVS['Sex'] = WVS['Sex'].fillna('Female') 
WVS['Age'] = WVS['Age'].fillna('30-49') 
WVS['Ethnicity'] = WVS['Ethnicity'].fillna('Turkish') 
WVS['Education'] = WVS['Education'].fillna('Complete secondary school: university-preparatory type') 

#### recoding DV: Vote for AKP 

def coding(col, codeDict):
  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded


AKP_bins = [0,1,2,3]
AKP_TYPE_labels = {"Vote AKP", "No Vote AKP", "Other"}              


WVS['Vote_AKP_Coded']= coding(WVS['Vote_AKP'], {'TR: AKP - Justice and Development Party': 0,
                    "TR: CHP - Republican Peoples Party": 1,
                    "Other": 2,
                    "TR: MHP - Nationalist Action Party -": 1,
                    "I would not vote": 2,
                    "No answer": 2,
                    "Don´t know": 2,
                    'TR: BDP (independent candidates for BDP, "Peace and Democracy Party")': 1})

WVS['Vote_AKP_labeled'] = pd.cut(WVS.Vote_AKP_Coded, AKP_bins, labels = AKP_TYPE_labels, right=False)

from matplotlib import pyplot
pyplot.hist(WVS.Vote_AKP_Coded)
pyplot.ylabel('Frequency')
pyplot.xlabel('Vote_AKP_Coded')
pyplot.title('Vote_AKP_Coded')
pyplot.show()

####### 

bins = [0,1,2]
labels = {"Satisfied", "Not Satisfied" }              


WVS['Economic_Satisfaction_Coded']= coding(WVS['Economic_Satisfaction'], {'5':0, '6':0, '7':0, 'Completely satisfied':0, 'Completely dissatisfied':1,
   '8':0, '9':0, '3':1, '4':1, 'Don´t know':1, '2':1, 'No answer; BH: Refused':1})

WVS['Economic_Satisfaction_Labeled'] = pd.cut(WVS.Economic_Satisfaction_Coded, AKP_bins, labels = AKP_TYPE_labels, right=False)


from matplotlib import pyplot
pyplot.hist(WVS.Economic_Satisfaction_Coded)
pyplot.ylabel('Frequency')
pyplot.xlabel('Economic_Satisfaction_Coded')
pyplot.title('Economic_Satisfaction_Coded')
pyplot.show()

######

bins = [0,1,2]
labels = {"Right", "Left" }              


WVS['Ideology_Coded']= coding(WVS['Ideology'], {'5':0, '6':0, '7':0, 'Right':0, 'Left':1,
   '8':0, '9':0, '3':1, '4':1, 'Don´t know':1, '2':1, 'No answer':1})

WVS['Ideology_labeled'] = pd.cut(WVS.Ideology_Coded, AKP_bins, labels = AKP_TYPE_labels, right=False)

from matplotlib import pyplot
pyplot.hist(WVS.Ideology_Coded)
pyplot.ylabel('Frequency')
pyplot.xlabel('Ideology_Coded')
pyplot.title('Ideology_Coded')
pyplot.show()

######

bins = [0,1,2]
labels = {"Religious", "Not Religious" }              


WVS['Religiousity_Coded']= coding(WVS['Religiousity'], {
        'Only on special holy days':0, 'More than once a week':0,
       'Once a week':0, 'Never, practically never':1, 'Once a month':0,
       'No answer':1, 'Less often':1, 'Once a year':1, 'Don´t know':1})

WVS['Religiousity_Labeled'] = pd.cut(WVS.Religiousity_Coded, AKP_bins, labels = AKP_TYPE_labels, right=False)


from matplotlib import pyplot
pyplot.hist(WVS.Religiousity_Coded)
pyplot.ylabel('Frequency')
pyplot.xlabel('Religiousity_Coded')
pyplot.title('Religiousity_Coded')
pyplot.show()

#### 

bins = [0,1,2]
labels = {"Employed", "Not Employed"}              


WVS['Employment_Coded']= coding(WVS['Employment'], {'Housewife':1, 'Full time':0, 'Retired':1, 'Students':1,
   'Self employed':0,'Unemployed':1, 'Other':1, 'Part time':0})

WVS['Employment_Labeled'] = pd.cut(WVS.Employment_Coded, AKP_bins, labels = AKP_TYPE_labels, right=False)

from matplotlib import pyplot
pyplot.hist(WVS.Employment_Coded)
pyplot.ylabel('Frequency')
pyplot.xlabel('Employment_Coded')
pyplot.title('Employment_Coded')
pyplot.show()

### 

bins = [0,1,2,3]
labels = {"Upper Class", "Middle Class", "Lower Class"}              

WVS['Social_Class_Coded']= coding(WVS['Social_Class'], {'Upper middle class':0, 'Lower middle class':2,
   'Working class':1,'Upper class':0, 'Lower class':2, 'No answer':1, 'Don´t know':1})

WVS['Social_Class_Labeled'] = pd.cut(WVS.Social_Class_Coded, AKP_bins, labels = AKP_TYPE_labels, right=False)

from matplotlib import pyplot
pyplot.hist(WVS.Social_Class_Coded)
pyplot.ylabel('Frequency')
pyplot.xlabel('Social_Class_Labeled')
pyplot.title('Social_Class_Labeled')
pyplot.show()

##### 

bins = [0,1,2,3]
labels = {"High", "Middle", "Low"}              

WVS['Income_Coded']= coding(WVS['Income'], {'Seventh step':0, 'Lower step':2, 'second step':2, 'Nineth step':0,
       'Fourth step':1, 'Fifth step':1, 'Third step':0, 'Sixth step':1,
       'Eigth step':0, 'Don´t know':1, 'No answer':1, 'Tenth step':0})

WVS['Income_Labeled'] = pd.cut(WVS.Income_Coded, AKP_bins, labels = AKP_TYPE_labels, right=False)

from matplotlib import pyplot
pyplot.hist(WVS.Income_Coded)
pyplot.ylabel('Frequency')
pyplot.xlabel('Income_Coded')
pyplot.title('Income_Coded')
pyplot.show()

##### 

bins = [0,1,2]
labels = {"Male", "Female"}              

WVS['Sex_Coded']= coding(WVS['Sex'], {'Female':1, 'Male':0})

WVS['Sex_Labeled'] = pd.cut(WVS.Sex_Coded, AKP_bins, labels = AKP_TYPE_labels, right=False)

from matplotlib import pyplot
pyplot.hist(WVS.Sex_Coded)
pyplot.ylabel('Frequency')
pyplot.xlabel('Sex_Coded')
pyplot.title('Sex_Coded')
pyplot.show()

##### Age

bins = [0,1,2]
labels = {'30-49', '50 and more', 'Up to 29'}              

WVS['Age_Coded']= coding(WVS['Age'], {'30-49':0, '50 and more':1, 'Up to 29':2})

WVS['Age_Labeled'] = pd.cut(WVS.Age_Coded, AKP_bins, labels = AKP_TYPE_labels, right=False)

from matplotlib import pyplot
pyplot.hist(WVS.Age_Coded)
pyplot.ylabel('Frequency')
pyplot.xlabel('Age_Coded')
pyplot.title('Age_Coded')
pyplot.show()

####

bins = [0,1,2]
labels = {'Turkish', 'Other'}              

WVS['Ethnicity_Coded']= coding(WVS['Ethnicity'], {'Turkish':0, 'Other':1,'Kurdish':1})

WVS['Ethnicity_Labeled'] = pd.cut(WVS.Ethnicity_Coded, AKP_bins, labels = AKP_TYPE_labels, right=False)

from matplotlib import pyplot
pyplot.hist(WVS.Ethnicity_Coded)
pyplot.ylabel('Frequency')
pyplot.xlabel('Ethnicity_Coded')
pyplot.title('Ethnicity_Coded')
pyplot.show()

##### 

bins = [0,1,2]
labels = {"High", "Middle", "Low"}              

WVS['Education_Coded']= coding(WVS['Education'], {'Complete primary school':2,
       'University - level education, with degree':0,
       'Incomplete secondary school: university-preparatory type':1,
       'Complete secondary school: university-preparatory type':1,
       'Some university-level education, without degree':0,
       'Incomplete secondary school: technical/ vocational type':1,
       'Complete secondary school: technical/ vocational type':1,
       'No formal education':2, 'Incomplete primary school':2})

WVS['Education_Labeled'] = pd.cut(WVS.Education_Coded, AKP_bins, labels = AKP_TYPE_labels, right=False)


from matplotlib import pyplot
pyplot.hist(WVS.Education_Coded)
pyplot.ylabel('Frequency')
pyplot.xlabel('Education_Coded')
pyplot.title('Education_Coded')
pyplot.show()

#### 
import statsmodels.formula.api as smf

WVS_OLS = smf.ols('Religiousity_Coded ~  Economic_Satisfaction_Coded + Ideology_Coded + Vote_AKP_Coded + Employment_Coded + Social_Class_Coded+ Income_Coded+ Sex_Coded+ Age_Coded + Ethnicity_Coded + Education_Coded', data=WVS).fit()
print(WVS_OLS.summary())
plt.rc('figure', figsize=(12, 7))
#plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
plt.text(0.01, 0.05, str(WVS_OLS.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.savefig('output.png')
plt.show() #### R-squared = .15 ### some insignificant ID P-Values 

#####

WVS.to_excel('WVS1.xlsx')

list(WVS.columns)


X = WVS.iloc[:, [13,15,17,19,21,23,25,27,29, 31]].values

y = WVS.iloc[:,[11]].values ### multiclass DV

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0,1,2,3,4,5,6,7,8,9])### features 0-8
X = onehotencoder.fit_transform(X).toarray()
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###################################Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV


classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
classifier.score(X_test,y_test)
y_pred = classifier.predict(X_test)# Predicting the Test set results


#define the model and parameters for Gridsearch 
knn = KNeighborsClassifier()

parameters = {'n_neighbors':[4,5,6,7],
              'leaf_size':[1,3,5],
              'algorithm':['auto', 'kd_tree'],
              'n_jobs':[-1]}


model = GridSearchCV(knn, param_grid=parameters)
model.fit(X_train,y_train.ravel())
model.score(X_train,y_train.ravel())
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train.ravel(), cv=5)
                                         

############################ Logistic Regression model

X = WVS.iloc[:, [13,15,17,19,21,23,25,27,29, 31]].values

y = WVS.iloc[:,[11]].values ### multiclass DV

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train, y_train)

regressor.fit(X_train,y_train)
regressor.score(X_test,y_test)### 0.52 

# Gridsearch for Logistics  
from sklearn import linear_model, datasets
logistic = linear_model.LogisticRegression()
penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
best_model = clf.fit(X_train, y_train.ravel())
best_model.score(X_train,y_train.ravel())
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
best_model.predict(X)



############################################### Random Forest

X = WVS.iloc[:, [13,15,17,19,21,23,25,27,29, 31]].values

y = WVS.iloc[:,[11]].values ### multiclass DV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
classifier.score(X_test,y_test)### score .48


rfc = RandomForestClassifier(n_estimators=10,criterion='gini',max_depth=4,random_state=0)
rfc.fit(X_train,y_train)
rfc.score(X_test,y_test)### .54


#Random Forest Gridsearch 

rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [50, 100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train.ravel())

CV_rfc.best_params_
rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 100, max_depth=6, criterion='gini')

rfc1.fit(X_train, y_train)
rfc1.score(X_train, y_train)


# Learning & Validation Curves 

from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge

train_scores, valid_scores = validation_curve(Ridge(), X, y, "alpha", np.logspace(-7, 3, 3), cv=5)
train_scores            
valid_scores  

from sklearn.model_selection import learning_curve
from sklearn.svm import SVC

train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv=5)
train_sizes            
train_scores           
valid_scores        

###### SVM with GridSearch 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
classifier.score(X_train, y_train)###Kernel = linear 

classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
classifier.score(X_train, y_train)###Kernel = rbf


from sklearn.model_selection import GridSearchCV
params = {
            'C':[0.1, 1, 10],
            'kernel':['linear', 'poly', 'rbf'],
            'degree':[2,3,4],
            'gamma':[0.001, 0.01, 0.1],
            'tol':[0.001, 0.01, 0.1]        
        }



from sklearn.svm import SVC
clf = SVC()
svm = GridSearchCV(clf,params)
svm.fit(X_train,y_train.ravel())
svm.best_score_#####0.52
svm.best_params_### {'C': 1, 'degree': 3, 'gamma': 0.001, 'kernel': 'poly', 'tol': 0.001}

svm_best = SVC(C = 10, degree= 2, gamma = 0.0001, kernel= 'rbf')
svm_best.fit(X_train,y_train)
y_pred = svm_best.predict(X_test)
svm_best.score(X_test,y_test)##0.527


#######################


from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(hidden_layer_sizes=(10,10), activation='logistic',
                   solver='sgd', max_iter = 10000)
nn.fit(X_train,y_train)
nn.score(X_train,y_train)### 0.40
