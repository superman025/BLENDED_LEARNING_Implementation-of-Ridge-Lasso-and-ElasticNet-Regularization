# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Bring in the necessary libraries such as NumPy, Pandas, Matplotlib, and scikit-learn
2. Load the Dataset: Load the car price dataset into your environment.
3. Data Preprocessing: Handle any missing data and encode categorical variables as needed.
4. Define Features and Target: Split the dataset into features (X) and the target variable (y), where the target variable is the car price.
5. Split Data: Divide the dataset into training and testing sets.
6. Build Ridge, Lasso, and ElasticNet Models: Initialize Ridge Regression, Lasso Regression, and      ElasticNet Regression models.
7. Train the Models: Fit the Ridge, Lasso, and ElasticNet models to the training data
8. Evaluate Performance: Assess the models’ performance using cross-validation or evaluation metrics such as Mean Squared Error (MSE) and R² score
9. Display Model Parameters: Output the coefficients and intercept for each model.
10. Make Predictions & Compare: Predict car prices using the trained models and compare the predicted values with the actual values.


## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: sri jai.v
RegisterNumber:  25018437
*/
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("encoded_car_data (1).csv")
data.head()

# Data preprocessing
data = pd.get_dummies(data, drop_first=True)

# Splitting the data into features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Standardizing the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1))

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the models
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('regressor', model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results[name] = {'MSE': mse, 'R2 Score': r2}

# Print results
print("Name:SRIJAI V")
print("Reg No:25018437")
for model_name, metrics in results.items():
    print(f"{model_name} - MSE: {metrics['MSE']:.2f}, R2 Score: {metrics['R2 Score']:.2f}")

# Visualization
results_df = pd.DataFrame(results).T.reset_index()
results_df.rename(columns={'index': 'Model'}, inplace=True)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MSE', data=results_df)
plt.title("Mean Squared Error (MSE)")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R2 Score', data=results_df)
plt.title("R2 Score")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
*/
```

## Output:
<img width="1388" height="648" alt="image" src="https://github.com/user-attachments/assets/b62898f0-c980-4307-8a94-06e95ceab5cd" />



## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
