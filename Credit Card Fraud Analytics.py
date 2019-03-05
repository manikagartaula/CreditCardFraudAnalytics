#!/usr/bin/env python
# coding: utf-8
#Author: Manika Gartaula
#Date: 3/2/2019

#Import necessary libraries to load and analyze the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read data from csv file
credit_data = pd.read_csv('creditcard.csv')

#Check data specifications
print(credit_data.shape)
print(credit_data.head(3))

#31 columns: 1 Target Variable (Class), 30 Predictor Variables
#Check for missing values
credit_data.isnull().values.any()

#Separate target and predictor variables
target = credit_data.iloc[:,-1]
predictors = credit_data.iloc[:,0:30]

#Check summary statistics on the predictor variables
predictors.describe()

#Check for distribution of target variable
print(target.value_counts())

#Visualize two interpretable predictors Time and Amount
#predictors['Amount'] = np.log(predictors.Amount + 1) #Highly skewed; but log didn't change prediction results

plt.hist(predictors['Time'])
plt.hist(predictors['Amount'])

#Predictor variables v1..v28 are result of PCA and are in standard form
#Standardize predictor variables Time and Amount to prevent bias
from sklearn import preprocessing
predictors['Time'] = preprocessing.scale(predictors['Time'])
predictors['Amount'] = preprocessing.scale(predictors['Amount'])
predictors.describe()

#Predictive Modeling on Unbalanced Data
#Split data into training and test data sets for predictive modeling
from sklearn.model_selection import train_test_split
p_train, p_test, t_train, t_test = train_test_split(predictors, target, test_size = 0.3, random_state = 101)

#Check statistics for train and test data sets
print(p_train.shape)
print(p_test.shape)

print(t_train.value_counts())
print(t_test.value_counts())

#Test different classification models on the data for accurate predictions: Logistic Regression and Random Forest
#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model1 = LogisticRegression()

#Train predictive model
model1.fit(p_train, t_train)

#Test predictive model
t_pred1 = model1.predict(p_test)

#Test accuracy of model
print(accuracy_score(t_test, t_pred1))

#99% accuracy doesn't say much because of the unbalanced target class
#Confusion matrix and roc curve depict better understanding of model

#Check classification matrix for the prediction results
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sn
names = ['Not Fraud', 'Fraud']
cm = confusion_matrix(t_test, t_pred1)

# Create pandas dataframe
dataframe = pd.DataFrame(cm, index=names, columns=names)
# Create heatmap
sn.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

print(classification_report(t_test, t_pred1, target_names = names))

#Visualize fraud transaction prediction accuracy using ROC curve
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds = roc_curve(t_test, t_pred1)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#Random Forest
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier()

#Train predictive model using training data set
model2.fit(p_train, t_train)

#Test predictive model using testing data set
t_pred2 = model2.predict(p_test)

#Test accuracy of model
print(accuracy_score(t_test, t_pred1))

#Check classification matrix for the prediction results
cm2 = confusion_matrix(t_test, t_pred2)

# Create pandas dataframe
dataframe = pd.DataFrame(cm2, index=names, columns=names)
# Create heatmap
sn.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

print(classification_report(t_test, t_pred2, target_names = names))

#Visualize fraud transaction prediction accuracy using ROC curve
fpr, tpr, thresholds = roc_curve(t_test, t_pred2)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#Random Forest works well with unbalanced data, and produces better predictions

#Test effect of data balancing on prediction using Logistic Regression - Over sampling the minority class

train_data, test_data = train_test_split(credit_data, test_size = 0.3, random_state = 101)
print(train_data.shape)
print(test_data.shape)

#Separate majority and minority class for training data
train_majority = train_data[train_data.Class == 0]
train_minority = train_data[train_data.Class == 1]
print(train_majority.shape)
print(train_minority.shape)

#Oversampling minority class
from sklearn.utils import resample
train_minority_upsampled = resample(train_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=199016,    # to match majority class
                                 random_state=123)
print(train_minority_upsampled.shape)


#Combine the majority and upsampled minority class and run Logistic Regression
train_sample_data = pd.concat([train_majority, train_minority_upsampled])
print(train_sample_data.Class.value_counts()) #balanced data

#Separate predictor and target variables
p2_train = train_sample_data.iloc[:,0:30]
t2_train = train_sample_data.iloc[:,-1]

p2_test = test_data.iloc[:,0:30]
t2_test = test_data.iloc[:,-1]
print(t2_test.value_counts())

#Run Logistic Regression model on balanced data
model3 = LogisticRegression()
model3.fit(p2_train, t2_train)

t_pred3 = model3.predict(p2_test)
print(accuracy_score(t2_test, t_pred3))


#Check confusion matric for prediction results
cm3 = confusion_matrix(t2_test, t_pred3)

# Create pandas dataframe
dataframe = pd.DataFrame(cm3, index=names, columns=names)
# Create heatmap
sn.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

print(classification_report(t2_test, t3_pred, target_names = names))


#Visualize fraud transaction prediction accuracy using ROC curve
fpr, tpr, thresholds = roc_curve(t2_test, t_pred3)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#The area under the curve is the highest for the balanced data
#Though the overall accuracy is least, the model is able to predict the minority class with more accuracy


