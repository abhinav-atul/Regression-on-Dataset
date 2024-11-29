import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('user_behavior_dataset.csv')

# Display the first 5 records
print("First 5 records of the dataset:")
print(data.head())

# Renaming columns for simplicity
data.columns = ['UserID', 'DeviceModel', 'OS', 'AppUsage', 'ScreenOnTime', 'BatteryDrain',
                'NumApps', 'DataUsage', 'Age', 'Gender', 'BehaviorClass']

# Features (X) and Target (y) for classification
X_class = data[['AppUsage', 'ScreenOnTime', 'BatteryDrain', 'NumApps', 'DataUsage']]
y_class = data['BehaviorClass']

# Splitting the data for classification (Logistic Regression and KNN)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.3, random_state=42)

# --- LOGISTIC REGRESSION ---
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_class, y_train_class)
y_pred_logistic = logistic_model.predict(X_test_class)

# Plot 1: Confusion Matrix for Logistic Regression
conf_matrix = confusion_matrix(y_test_class, y_pred_logistic)
plt.figure(figsize=(6, 5))
plt.matshow(conf_matrix, cmap='coolwarm', fignum=1)
plt.title("Logistic Regression - Confusion Matrix")
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# --- KNN (K-NEAREST NEIGHBORS) ---
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_class, y_train_class)
y_pred_knn = knn_model.predict(X_test_class)

# Plot 2: Accuracy vs. Number of Neighbors (K)
k_values = range(1, 21)
knn_accuracies = []
for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_class, y_train_class)
    knn_accuracies.append(knn_model.score(X_test_class, y_test_class))

plt.figure(figsize=(8, 5))
plt.plot(k_values, knn_accuracies, marker='o', linestyle='--', color='b')
plt.title("KNN - Accuracy vs. Number of Neighbors")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

# --- LINEAR REGRESSION ---
# Linear regression with AppUsage as the target and BatteryDrain as a feature
X_reg = data[['BatteryDrain']]
y_reg = data['AppUsage']

# Splitting the data for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Fit the model
linear_model = LinearRegression()
linear_model.fit(X_train_reg, y_train_reg)

# Predict
y_pred_linear = linear_model.predict(X_test_reg)

# Ensure shapes match for plotting
y_test_reg = y_test_reg.reset_index(drop=True)  # Reset index of y_test_reg to align with y_pred_linear

# Plot 3: Actual vs. Predicted Values (Linear Regression)
plt.figure(figsize=(8, 5))
plt.scatter(y_test_reg, y_pred_linear, color='blue', alpha=0.6)
plt.title("Linear Regression - Actual vs. Predicted")
plt.xlabel("Actual App Usage (min/day)")
plt.ylabel("Predicted App Usage (min/day)")
plt.grid()
plt.show()

# Plot 4: Residuals Histogram (Linear Regression)
residuals = y_test_reg - y_pred_linear  # Residuals calculation
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=20, color='purple', alpha=0.7)
plt.title("Linear Regression - Residuals Histogram")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid()
plt.show()
