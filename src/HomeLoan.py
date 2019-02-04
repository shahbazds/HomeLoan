
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

#Confusion Matrix Function
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
   
    import numpy as np
    import itertools
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()






dfLoanTrain = pd.read_csv('train_Loan_Home.csv')
dfLoanTrain.isna().sum()
dfLoanTrain.info()
dfLoanTrain.hist()

dfLoanTrain['Gender'] = dfLoanTrain['Gender'].fillna( dfLoanTrain['Gender'].dropna().mode().values[0] )
dfLoanTrain['Married'] = dfLoanTrain['Married'].fillna( dfLoanTrain['Married'].dropna().mode().values[0] )
dfLoanTrain['Dependents'] = dfLoanTrain['Dependents'].fillna( dfLoanTrain['Dependents'].dropna().mode().values[0] )
dfLoanTrain['Self_Employed'] = dfLoanTrain['Self_Employed'].fillna( dfLoanTrain['Self_Employed'].dropna().mode().values[0] )
dfLoanTrain['LoanAmount'] = dfLoanTrain['LoanAmount'].fillna( dfLoanTrain['LoanAmount'].dropna().mean() )
dfLoanTrain['Loan_Amount_Term'] = dfLoanTrain['Loan_Amount_Term'].fillna( dfLoanTrain['Loan_Amount_Term'].dropna().mode().values[0] )
dfLoanTrain['Credit_History'] = dfLoanTrain['Credit_History'].fillna( dfLoanTrain['Credit_History'].dropna().mode().values[0] )
dfLoanTrain['Dependents'] = dfLoanTrain['Dependents'].str.rstrip('+')
dfLoanTrain['Gender'] = dfLoanTrain['Gender'].map({'Female':0,'Male':1}).astype(np.int)
dfLoanTrain['Married'] = dfLoanTrain['Married'].map({'No':0, 'Yes':1}).astype(np.int)
dfLoanTrain['Education'] = dfLoanTrain['Education'].map({'Not Graduate':0, 'Graduate':1}).astype(np.int)
dfLoanTrain['Self_Employed'] = dfLoanTrain['Self_Employed'].map({'No':0, 'Yes':1}).astype(np.int)
dfLoanTrain['Loan_Status'] = dfLoanTrain['Loan_Status'].map({'N':0, 'Y':1}).astype(np.int)
dfLoanTrain['Property_Area'] = dfLoanTrain['Property_Area'].map({'Urban':0, 'Rural':1,'Semiurban' :2}).astype(np.int)
dfLoanTrain['Dependents'] = dfLoanTrain['Dependents'].astype(np.int)

dfLoanTrain.isna().sum()


#Plotting Correlation
corr = dfLoanTrain.iloc[:,1:].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#Seperating Target and Features
X = dfLoanTrain.iloc[:, 1:-1]
y = dfLoanTrain.iloc[:,-1]   
 
#Splitting Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25, random_state=0)

#fitting Descision Tree
from sklearn.ensemble import RandomForestClassifier
classifierR = RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=0,max_depth=4)
classifierR.fit(X_train, y_train)

#Prediction
y_pred = classifierR.predict(X_test)

#confusion Matrix and Score
from sklearn.metrics import confusion_matrix

print(classifierR.score(X_test,y_test))
cms = confusion_matrix(y_test,y_pred)
type(cms)
print(cms)
plot_confusion_matrix(cm           = cms, 
                      target_names = ['high','low'],
                      normalize    = False,
                      title        = "Confusion Matrix for Random Forest")
#K-Fold Accuracy Check
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifierR,X,y,cv=10,scoring='accuracy')
print("Average K-Fold Accuracy: ", scores.mean())


#Logistic Regression

from sklearn.linear_model import LogisticRegression
classifierl = LogisticRegression()
classifierl.fit(X_train, y_train)

y_pred = classifierl.predict(X_test)

#confusion Matrix and Score
from sklearn.metrics import confusion_matrix

print(classifierl.score(X_test,y_test))
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cm           = cm, 
                      target_names = ['high','low'],
                      normalize    = False,
                      title        = "Confusion Matrix for Logistic Regression")

#K-Fold Accuracy Check
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifierl,X,y,cv=10,scoring='accuracy')
print("Average K-Fold Accuracy: ", scores.mean())


#fitting k-NN to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p=2)
classifier.fit(X_train,y_train)



#Predicting the test set results
y_pred = classifier.predict(X_test)

#Creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cm           = cm, 
                      target_names = ['high','low'],
                      normalize    = False,
                      title        = "Confusion Matrix")


#accuracy
print(classifier.score(X_test,y_test))

#K-Fold Accuracy Check
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier,X,y,cv=10,scoring='accuracy')
print("Average K-Fold Accuracy: ", scores.mean())



#Test Data
dfLoanTest = pd.read_csv('test_Loan_Home.csv')

#Data Cleaning
dfLoanTest['Gender'] = dfLoanTest['Gender'].fillna( dfLoanTest['Gender'].dropna().mode().values[0] )
dfLoanTest['Married'] = dfLoanTest['Married'].fillna( dfLoanTest['Married'].dropna().mode().values[0] )
dfLoanTest['Dependents'] = dfLoanTest['Dependents'].fillna( dfLoanTest['Dependents'].dropna().mode().values[0] )
dfLoanTest['Self_Employed'] = dfLoanTest['Self_Employed'].fillna( dfLoanTest['Self_Employed'].dropna().mode().values[0] )
dfLoanTest['LoanAmount'] = dfLoanTest['LoanAmount'].fillna( dfLoanTest['LoanAmount'].dropna().mean() )
dfLoanTest['Loan_Amount_Term'] = dfLoanTest['Loan_Amount_Term'].fillna( dfLoanTest['Loan_Amount_Term'].dropna().mode().values[0] )
dfLoanTest['Credit_History'] = dfLoanTest['Credit_History'].fillna( dfLoanTest['Credit_History'].dropna().mode().values[0] )
dfLoanTest['Dependents'] = dfLoanTest['Dependents'].str.rstrip('+')
dfLoanTest['Gender'] = dfLoanTest['Gender'].map({'Female':0,'Male':1}).astype(np.int)
dfLoanTest['Married'] = dfLoanTest['Married'].map({'No':0, 'Yes':1}).astype(np.int)
dfLoanTest['Education'] = dfLoanTest['Education'].map({'Not Graduate':0, 'Graduate':1}).astype(np.int)
dfLoanTest['Self_Employed'] = dfLoanTest['Self_Employed'].map({'No':0, 'Yes':1}).astype(np.int)
dfLoanTest['Dependents'] = dfLoanTest['Dependents'].astype(np.int)
dfLoanTest['Property_Area'] = dfLoanTest['Property_Area'].fillna( dfLoanTest['Property_Area'].dropna().mode().values[0] )
dfLoanTest['Property_Area'] = dfLoanTest['Property_Area'].map({'Urban':0, 'Rural':1,'Semiurban' :2}).astype(np.int)


X_Test = dfLoanTest.iloc[:, 1:]
#Random Forest
result = classifierR.predict(X_Test)
dfLoanTest['Loan_Status'] = result





