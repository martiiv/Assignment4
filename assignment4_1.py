from sklearn import svm # import svm
from sklearn import metrics # Evaluation metrics 

# Basics 
import numpy as np              # Math
import pandas as pd             # Dataframe
import matplotlib.pyplot as plt # Plotting

#* Task 1 Find an implementation of the SVM model that offers a capability to adjust key parameters of the model.
# We are doing binary classification, we will implement a simple support vector machine usin scikit-learn
train = pd.read_csv('hw4Train.csv')    # Read the train data
test = pd.read_csv('hw4Test.csv')      # Read the test data

# Splitting data 
trainX = train.drop('class', axis=1)    # Drop the class column
trainY = train['class']                 # Get the class column

testX = test.drop('class', axis=1)      # Drop the class column
testY = test['class']                   # Get the class column

# Plotting data
x = train["Sepal Length"]                                      #Defining x attribute 
y = train["Petal Length"]                                      #Defining y attribute
attrClass = train["class"]

colors = {"Iris-versicolor":"red", "Iris-virginica":"blue"}     #We map the colors to the different classtypes from the csv file 
fig, ax = plt.subplots(figsize=(8,8))                           #Defining figure and labeling  

plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.scatter(x, y, c=attrClass.map(colors))                  #Plotting the scatterplot

plt.show()                                              

#Defining classifier
classifier = svm.SVC(kernel='rbf', C=1000) # Create a classifier with a linear kernel and penalty parameter =1
classifier.fit(trainX, trainY)              # Fit the classifier to the training data
predictions = classifier.predict(testX)                    # Predict the test data

# Classifier done evaluating results:
print("Accuracy:",metrics.accuracy_score(testY, predictions)) # Print the accuracy score
