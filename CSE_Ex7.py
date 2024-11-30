import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report

data = pd.read_csv('user_behavior_dataset.csv')
data.columns = ['UserID', 'DeviceModel', 'OS', 'AppUsage', 'ScreenOnTime', 'BatteryDrain','NumApps', 'DataUsage', 'Age', 'Gender', 'BehaviorClass']

print(data.columns)

X_class = data[['AppUsage', 'ScreenOnTime', 'BatteryDrain', 'NumApps', 'DataUsage']]
y_class = data['BehaviorClass']

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.3, random_state=42)

# --- LOGISTIC REGRESSION ---
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_class, y_train_class)
y_pred_logistic = logistic_model.predict(X_test_class)

accuracy_logistic = accuracy_score(y_test_class, y_pred_logistic)
print(f"Accuracy for Logistic Regression: {accuracy_logistic:.2f}")

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

accuracy_knn_5 = accuracy_score(y_test_class, y_pred_knn)
print(f"Accuracy for KNN with K=5: {accuracy_knn_5:.2f}")

# Plot 2: Accuracy vs. Number of Neighbors (K)
k_values = range(1, 21)
knn_accuracies = []
for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_class, y_train_class)
    acc = knn_model.score(X_test_class, y_test_class)
    knn_accuracies.append(acc)
    print(f"Accuracy for K={k}: {acc:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(k_values, knn_accuracies, marker='o', linestyle='--', color='b')
plt.title("KNN - Accuracy vs. Number of Neighbors")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

# --- LINEAR REGRESSION ---

X_reg = data[['BatteryDrain']]
y_reg = data['AppUsage']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train_reg, y_train_reg)

y_pred_linear = linear_model.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_linear)
print(f"Mean Squared Error for Linear Regression: {mse:.2f}")
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE) for Linear Regression: {rmse:.2f}")

y_test_reg = y_test_reg.reset_index(drop=True)  

# Plot 3: Actual vs. Predicted Values (Linear Regression)
plt.figure(figsize=(8, 5))
plt.scatter(y_test_reg, y_pred_linear, color='blue', alpha=0.6)
plt.title("Linear Regression - Actual vs. Predicted")
plt.xlabel("Actual App Usage (min/day)")
plt.ylabel("Predicted App Usage (min/day)")
plt.grid()
plt.show()

# Plot 4: Residuals Histogram (Linear Regression)
residuals = y_test_reg - y_pred_linear 
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=20, color='purple', alpha=0.7)
plt.title("Linear Regression - Residuals Histogram")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid()
plt.show()
