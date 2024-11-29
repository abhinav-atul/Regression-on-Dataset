# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

# Read the dataset from a CSV file
data = pd.read_csv('user_behavior_dataset.csv')

# Display the first few rows of the dataset
print(data.head())

# Assuming the dataset has columns 'App Usage Time (min/day)' and 'Age' for linear regression
X_linear = data[['App Usage Time (min/day)']].values  # Independent variable(s)
y_linear = data['Age'].values                          # Dependent variable

# Train-test split for linear regression
X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train_linear, y_train_linear)

# Predictions
y_pred_linear = linear_model.predict(X_test_linear)

# Plotting Linear Regression Results
plt.figure(figsize=(10, 5))
plt.scatter(X_test_linear, y_test_linear, color='blue', label='Actual data')
plt.plot(X_test_linear, y_pred_linear, color='red', linewidth=2, label='Predicted line')
plt.title('Linear Regression: App Usage Time vs Age')
plt.xlabel('App Usage Time (min/day)')
plt.ylabel('Age')
plt.legend()
plt.show()

# Print Mean Squared Error for Linear Regression
mse = mean_squared_error(y_test_linear, y_pred_linear)
print(f'Mean Squared Error (Linear Regression): {mse:.2f}')

# For KNN and Logistic Regression: Assuming columns 'App Usage Time (min/day)', 'Age', and 'Target'
X_classification = data[['App Usage Time (min/day)', 'Age']].values  # Independent variables for classification
y_classification = data['Age'].values                               # Binary target variable

# Train-test split for classification tasks
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_classification, y_train_classification)

# Predictions for Logistic Regression
y_pred_logistic = logistic_model.predict(X_test_classification)

# Accuracy for Logistic Regression
logistic_accuracy = accuracy_score(y_test_classification, y_pred_logistic)
print(f'Accuracy (Logistic Regression): {logistic_accuracy:.2f}')

# K-Nearest Neighbors Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_classification, y_train_classification)

# Predictions for KNN
y_pred_knn = knn_model.predict(X_test_classification)

# Accuracy for KNN
knn_accuracy = accuracy_score(y_test_classification, y_pred_knn)
print(f'Accuracy (KNN): {knn_accuracy:.2f}')
