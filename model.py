# importing required modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import cv2 as cv
import sklearn

# Load folder names (classes) from training directory
folders = []
DIR = r'./train'
for folder in os.listdir(DIR):
    folders.append(folder)
print(folders)

classes = ['No Glaucoma', 'Glaucoma']

train_labels = []
train_features = []
for folder in folders:
    path = os.path.join(DIR, folder)
    for image in os.listdir(path):
        if folder == 'class0':
            train_labels.append(0)  # Label for 'No Glaucoma'
        else:
            train_labels.append(1)  # Label for 'Glaucoma'
        image_path = os.path.join(path, image)
        img = cv.imread(image_path)
        gray = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)  # Convert to grayscale
        resized = cv.resize(gray, (290, 290))  # Resize image to 290x290
        train_features.append(resized)
print(train_features)
print(train_labels)

# Convert training features and labels to numpy arrays
x_train = np.array(train_features)
print(x_train.shape)
y_train = np.array(train_labels)
print(y_train.shape)

# Load testing data (images and labels)
test_features = []
test_labels = []
DIR = r'./test'
for folder in folders:
    path = os.path.join(DIR, folder)
    for image in os.listdir(path):
        if folder == 'class0':
            test_labels.append(0)  # Label for 'No Glaucoma'
        else:
            test_labels.append(1)  # Label for 'Glaucoma'
        image_path = os.path.join(path, image)
        img = cv.imread(image_path)
        gray = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)  # Convert to grayscale
        resized = cv.resize(gray, (290, 290))  # Resize image to 290x290
        test_features.append(resized)
print(test_features)
print(test_labels)

# Convert testing features and labels to numpy arrays
x_test = np.array(test_features)
print(x_test.shape)
y_test = np.array(test_labels)
print(y_test.shape)

# Function to plot a sample image and its label
def plot_sample(x, y, index):
    plt.imshow(x[index])
    plt.ylabel(classes[y[index]])
    plt.show()

# Scale the data by reshaping and normalizing pixel values
scaled_x_train = x_train.reshape(-1, 290 * 290) / 255
print(scaled_x_train.shape)
scaled_x_test = x_test.reshape(-1, 290 * 290) / 255
print(scaled_x_test.shape)

# Print first sample of scaled training and testing data
print(scaled_x_train[0])
print(scaled_x_test[0])

from sklearn.linear_model import LogisticRegression

# Creating a logistic regression model and training it
model = LogisticRegression(max_iter=1000)
model.fit(scaled_x_train, y_train)
print(model.score)

plot_sample(x_train, y_train, 352)

# Predict for a single sample from the training set
predicted = model.predict(scaled_x_train[[352]])
print(predicted[0])

# Predict labels for the test set
y_predictions = model.predict(scaled_x_test)
y_predictions.shape

from sklearn.metrics import classification_report, confusion_matrix

# Generate confusion matrix and plot heatmap
cm = confusion_matrix(y_predictions, y_test)

sns.heatmap(cm, annot=True)
plt.xlabel("Prediction")
plt.ylabel("Truth")
plt.show()

# Calculate and print true positives, false negatives, false positives, and true negatives
true_positive = cm[0][0]
false_negative = cm[0][1]
false_positive = cm[1][0]
true_negative = cm[1][1]
print("True positive: ", true_positive)
print("False negative: ", false_negative)
print("False positive: ", false_positive)
print("True negative: ", true_negative)

# Print classification report
print(classification_report(y_test, y_predictions))

# Function to predict and plot an image given its file path
def predict_and_plot(path):
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv.resize(gray, (290, 290))  # Resize image to 290x290
    x_var = np.array(resized)
    x_var = x_var.reshape(-1, 290 * 290) / 255  # Normalize pixel values
    predicted = model.predict(x_var)
    plt.imshow(img)
    plt.title(classes[predicted[0]])  # Display predicted class name
    plt.show()

# Test the predict_and_plot function
path = r'./test/class0/BEH-203.png'
predict_and_plot(path)

# Function to run a test with a different image
def new_func():
    path = r'./train/class1/BEH-68.png'
    predict_and_plot(path)

new_func()

# Importing various classification models
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

# Store models in a dictionary
models = {
    'svm': {
        'model': SVC()
    },
    'random_forest': {
        'model': RandomForestClassifier()
    },
    'logistic_regression': {
        'model': LogisticRegression(max_iter=1000)
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB()
    },
    'naive_bayes_multinomial': {
        'model': MultinomialNB()
    },
    'decision_tree': {
        'model': DecisionTreeClassifier()
    }
}

# Iterate over models and evaluate their performance
results = []
for model_name, classifier in models.items():
    model = classifier['model']
    model.fit(scaled_x_train, y_train)  # Train the model
    accuracy = model.score(scaled_x_test, y_test)  # Calculate accuracy
    y_predicted = model.predict(scaled_x_test)  # Predict test labels
    cm = confusion_matrix(y_test, y_predicted)  # Confusion matrix

    # Calculate precision, recall, and F1 score
    true_positive = cm[0][0]
    false_negative = cm[0][1]
    false_positive = cm[1][0]
    true_negative = cm[1][1]
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(model_name)

    # Store model results
    results.append({
        "Model name": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 score": f1_score
    })

# Convert results to DataFrame and print
data = pd.DataFrame(results)
data

results

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Initialize AdaBoost with DecisionTree as the base estimator
adaBoost = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=10)

# Fit the model
adaBoost.fit(scaled_x_train, y_train)

adaBoost.score(scaled_x_test, y_test)  # Print AdaBoost accuracy

# Save the AdaBoost model to a file
from joblib import dump
dump(adaBoost, "./")

