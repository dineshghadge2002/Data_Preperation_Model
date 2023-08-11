# Machine Learning Model Selection and Evaluation

This repository contains a Python script that performs data preprocessing, model selection, and evaluation for various machine learning tasks. The script covers different scenarios such as binary classification, multi-class classification, and linear regression.

## Overview

This script demonstrates the following steps:

1. **Data Loading:** Load a dataset from a CSV file provided by the user.

2. **Data Preprocessing:**
   - Handle categorical variables by creating dummy variables.
   - Select relevant features based on correlation with the target variable.
   - Handle null values by filling them with the mean of each column.

3. **Model Selection and Evaluation:**
   - Fit different types of regression models based on the dataset characteristics and target variable.
   - Perform model evaluation using appropriate metrics for each task:
     - Binary Classification: Logistic Regression
     - Multi-Class Classification: Multinomial Logistic Regression
     - Linear Regression: Simple Linear Regression and Multi-Linear Regression

## Prerequisites

- Python (>= 3.6)
- Libraries: pandas, matplotlib, seaborn, numpy, scikit-learn

## Getting Started

1. Clone the repository to your local machine:
   ```
   git clone https://github.com/dineshghadge2002/Data_Preperation_Model.git
   cd machine-learning-model-evaluation
   ```

2. Make sure you have the required libraries installed. You can install them using the following command:
   ```
   pip install pandas matplotlib seaborn numpy scikit-learn
   ```

3. Download a dataset in CSV format and place it in the project directory.

4. Run the Python script:
   ```
   python model_evaluation.py
   ```

## Results

The script will output model coefficients, intercepts, and relevant evaluation metrics (such as accuracy and mean absolute error) based on the type of task and dataset.

## Author

Dinesh Ghadge

---
