from sklearn import svm # import svm
from sklearn import metrics # Evaluation metrics 

# Basics 
import numpy as np              # Math
import pandas as pd             # Dataframe
import matplotlib.pyplot as plt # Plotting

#* Task 1 Find an implementation of the SVM model that offers a capability to adjust key parameters of the model.
# We are doing binary classification, we will implement a simple support vector machine usin scikit-learn
train = pd.read_csv('how4Train.csv')    # Read the train data
test = pd.read_csv('how4Test.csv')      # Read the test data

# Splitting data 
trainX = train.drop('Class', axis=1)    # Drop the class column
trainY = train['Class']                 # Get the class column

testX = test.drop('Class', axis=1)      # Drop the class column
testY = test['Class']                   # Get the class column

#Defining classifier
classifier = svm.SVC(kernel='linear', C=1) # Create a classifier with a linear kernel and penalty parameter =1
classifier.fit(trainX, trainY)              # Fit the classifier to the training data
predictions = classifier.predict(testX)                    # Predict the test data

# Classifier done evaluating results:
print("Accuracy:",metrics.accuracy_score(testY, predictions)) # Print the accuracy score
print("Precision:",metrics.precision_score(testY, predictions)) # Print the precision score
print("Recall:",metrics.recall_score(testY, predictions)) # Print the recall score




