# User Behavior Analysis with Linear and Classification Models

This repository contains Python code for analyzing user behavior data using different machine learning models, including Linear Regression, Logistic Regression, and K-Nearest Neighbors (KNN). The code demonstrates the process of training and evaluating these models on a dataset with features such as app usage time and user age.

## Repository Link
[Regression on Dataset](https://github.com/abhinav-atul/Regression-on-Dataset.git)

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Data Description](#data-description)
- [How to Run the Code](#how-to-run-the-code)
- [Code Explanation](#code-explanation)
- [Model Results](#model-results)
- [License](#license)

## Overview
The code in this repository is designed to:
- **Perform Linear Regression** to predict user age based on app usage time.
- **Apply Logistic Regression** and **K-Nearest Neighbors (KNN)** for classification tasks with the same dataset.

## Requirements
Ensure the following Python libraries are installed to run the code:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

You can install them via pip if not already installed:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Data Description
The code assumes the dataset `user_behavior_dataset.csv` includes the following columns:
- **'App Usage Time (min/day)'**: The amount of time spent using an app per day (independent variable).
- **'Age'**: The age of the user (target variable for linear regression and classification).
- **'Target'**: A binary classification target variable (for use in logistic regression and KNN).

## How to Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/abhinav-atul/Regression-on-Dataset.git
   cd Regression-on-Dataset
   ```
2. Place the `user_behavior_dataset.csv` file in the root directory.
3. Run the script using Python:
   ```bash
   python user_behavior_analysis.py
   ```

## Code Explanation

### Linear Regression
- **Goal**: Predict user age based on app usage time.
- **Approach**: 
  - Train a linear regression model using `App Usage Time (min/day)` as the independent variable.
  - Visualize the regression line and calculate the Mean Squared Error (MSE) to evaluate model performance.

### Logistic Regression and KNN
- **Goal**: Classify users into age categories (assuming `Target` is binary).
- **Approach**: 
  - Split data into training and test sets.
  - Train logistic regression and KNN models using `App Usage Time (min/day)` and `Age` as features.
  - Evaluate accuracy using the test data.

### Visualizations and Metrics
- The script generates:
  - A scatter plot of the actual vs. predicted values for linear regression.
  - Accuracy scores for both logistic regression and KNN models.
  - The Mean Squared Error for the linear regression model.

## Model Results
- **Linear Regression**:
  - MSE is printed to assess prediction accuracy.
- **Logistic Regression and KNN**:
  - Accuracy scores are printed to evaluate classification performance.
