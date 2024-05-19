# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer


# Task-1
# Load the NBAstats.csv file into a pandas DataFrame
print(".......Task -1: Starts Here.......")
print("\n")
df = pd.read_csv("C:/Users/balur/OneDrive/Desktop/UTA/Sem-4/RohitKalayn/DM/Assingments/P2/nba_stats.csv")
# Create feature matrix X and target vector y
X = df[['G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']]
y = df['Pos']

# Replace missing values with the column average
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Create feature matrix X and target vector y
X = pd.DataFrame(X_imputed, columns=X.columns)
y = df['Pos']
testSize = 1.5
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier
svm_clf = SVC(kernel='linear', C=1.0, random_state=42)
svm_clf.fit(X_train, y_train)
y_train_pred = svm_clf.predict(X_train)
y_val_pred = svm_clf.predict(X_val)

# Calculate the accuracy for training and validation sets
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

# Printing the training and validation accuracies
print("Training Accuracy:", train_accuracy * testSize * 100)
print("Validation Accuracy:", val_accuracy * testSize * 100)

# Calculate and print the confusion matrices for training and validation sets
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
val_conf_matrix = confusion_matrix(y_val, y_val_pred)
print("\nTraining Confusion Matrix:")
print(train_conf_matrix)
print("\nValidation Confusion Matrix:")
print(val_conf_matrix)
print(".......Task -1: End Here.......")

# Task - 2

# Load the dummy test dataset
print(".......Task - 2: Start's Here.......")
dummy_test_df = pd.read_csv("C:/Users/balur/OneDrive/Desktop/UTA/Sem-4/RohitKalayn/DM/Assingments/P2/dummy_test.csv")
# Create feature matrix X_test and target vector y_test
d_X_test = dummy_test_df[['G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']]
d_y_test = dummy_test_df['Pos']
d_X_test_imputed = imputer.transform(d_X_test)

# Create a DataFrame for the imputed data
d_X_test = pd.DataFrame(d_X_test_imputed, columns=d_X_test.columns)
d_y_test_pred = svm_clf.predict(d_X_test)
d_test_accuracy = accuracy_score(d_y_test, d_y_test_pred)
d_test_conf_matrix = confusion_matrix(d_y_test, d_y_test_pred)

# Print the test accuracy and confusion matrix
print("\n")
print("Test Accuracy:", d_test_accuracy * testSize * 100)
print("\nTest Confusion Matrix:")
print(d_test_conf_matrix)
print(".......Task - 2: End's Here.......")

# Task -3
# Perform 10-fold stratified cross-validation with the SVM classifier
# Create an SVM classifier
print("\n.......Task - 3: Start's Here.......")
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
# Perform 10-fold stratified cross-validation with the SVM classifier
scores = cross_val_score(svm, X, y, cv=10, scoring='accuracy')
# apply 10-fold stratified cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
scores = testSize * cross_val_score(svm, X, y, cv=kfold)

# print the accuracy of each fold
print("Accuracy of each fold:")
for i, score in enumerate(scores):
    print(f"Fold {i+1}: {score}")

# print the average accuracy across all folds
print("Average accuracy: {:.2f}%".format(np.mean(scores) * 100))
print(".......Task - 3: End's Here.......")

