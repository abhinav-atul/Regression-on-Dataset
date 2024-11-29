import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

# Read the dataset from a CSV file
data = pd.read_csv('user_behavior_dataset.csv')

# Strip whitespace from column names (if any)
data.columns = data.columns.str.strip()

# Prepare features and target variable for Linear Regression
X_linear = data[['App Usage Time (min/day)']].values  # Independent variable
y_linear = data['Age'].values                          # Dependent variable

# Train-test split for Linear Regression
X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train_linear, y_train_linear)

# Predictions for Linear Regression
y_pred_linear = linear_model.predict(X_test_linear)

# Mean Squared Error for Linear Regression
mse = mean_squared_error(y_test_linear, y_pred_linear)
print(f'Mean Squared Error (Linear Regression): {mse:.2f}')

# Prepare features and target variable for Logistic Regression and KNN
X_classification = data[['App Usage Time (min/day)', 'Age']].values  # Independent variables
y_classification = data['Screen On Time (hours/day)'].values   # Binary target variable

# Train-test split for classification tasks
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Scale the features for Logistic Regression and KNN
scaler = StandardScaler()
X_train_classification_scaled = scaler.fit_transform(X_train_classification)
X_test_classification_scaled = scaler.transform(X_test_classification)

# Logistic Regression Model
logistic_model = LogisticRegression(max_iter=500)
logistic_model.fit(X_train_classification_scaled, y_train_classification)

# Predictions for Logistic Regression
y_pred_logistic = logistic_model.predict(X_test_classification_scaled)

# Accuracy for Logistic Regression
logistic_accuracy = accuracy_score(y_test_classification, y_pred_logistic)
print(f'Accuracy (Logistic Regression): {logistic_accuracy:.2f}')

# K-Nearest Neighbors Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_classification_scaled, y_train_classification)

# Predictions for KNN
y_pred_knn = knn_model.predict(X_test_classification_scaled)

# Accuracy for KNN
knn_accuracy = accuracy_score(y_test_classification, y_pred_knn)
print(f'Accuracy (KNN): {knn_accuracy:.2f}')

# Classification report for Logistic Regression
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test_classification, y_pred_logistic))