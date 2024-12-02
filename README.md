# User Behavior Prediction and Analysis

This repository contains a Python project for analyzing and predicting user behavior based on a dataset of user activity metrics. The project demonstrates the use of machine learning algorithms for both classification and regression tasks.

## Overview

The code performs the following tasks:
1. **Data Preparation**:
   - Reads the dataset from a CSV file.
   - Splits the dataset into features and target variables for classification and regression tasks.

2. **Classification**:
   - Implements Logistic Regression and K-Nearest Neighbors (KNN) algorithms to predict user behavior class.
   - Evaluates model performance using metrics like accuracy and confusion matrix.

3. **Regression**:
   - Implements Linear Regression to predict app usage based on battery drain.
   - Evaluates model performance using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

## Prerequisites

Ensure you have the following installed:
- Python (>=3.7)
- Required libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Dataset

The dataset should be named `user_behavior_dataset.csv` and should be placed in the root directory. The expected columns in the dataset are:
- `AppUsage`: Average daily app usage (in minutes).
- `ScreenOnTime`: Average daily screen-on time (in minutes).
- `BatteryDrain`: Average daily battery drain percentage.
- `NumApps`: Number of installed applications.
- `DataUsage`: Average daily data usage (in MB).
- `BehaviorClass`: Class label indicating user behavior.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/abhinav-atul/Regression-on-Dataset
   cd Regression-on-Dataset
   ```

2. Run the script:
   ```bash
   python CSE_Ex7.py
   ```

## Outputs

### Classification
- Accuracy of Logistic Regression and KNN classifiers.
- Confusion Matrix plot for Logistic Regression.
- Accuracy vs. Number of Neighbors plot for KNN.

### Regression
- Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for Linear Regression.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for improvements or feature requests.

