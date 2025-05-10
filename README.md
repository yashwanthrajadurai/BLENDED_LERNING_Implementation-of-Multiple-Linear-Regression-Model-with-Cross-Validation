# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices
# Name : YASHWANTH RAJA DURAI.V
# REG NO : 212222040284
## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**:  
   Bring in the necessary libraries.

2. **Load the Dataset**:  
   Load the dataset into your environment.

3. **Data Preprocessing**:  
   Handle any missing data and encode categorical variables as needed.

4. **Define Features and Target**:  
   Split the dataset into features (X) and the target variable (y).

5. **Split Data**:  
   Divide the dataset into training and testing sets.

6. **Build Multiple Linear Regression Model**:  
   Initialize and create a multiple linear regression model.

7. **Train the Model**:  
   Fit the model to the training data.

8. **Evaluate Performance**:  
   Assess the model's performance using cross-validation.

9. **Display Model Parameters**:  
   Output the model’s coefficients and intercept.

10. **Make Predictions & Compare**:  
    Predict outcomes and compare them to the actual values. 

## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: rohith v
RegisterNumber: 212223040174
*/
# Importing necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv")

# Data preprocessing
# Dropping unnecessary columns and handling categorical variables
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)

# Splitting the data into features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the model
model = LinearRegression()

# Fitting the model on the training data
model.fit(X_train, y_train)

# Evaluating model performance using cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

# Printing cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())

# Print model coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Make predictions
predictions = model.predict(X_test)

# Visualizing actual vs predicted prices
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect prediction line
plt.show()

```

## Output:
<img width="801" alt="Screenshot 2024-10-06 at 8 53 33 PM" src="https://github.com/user-attachments/assets/1a84f7be-ffb0-4073-b864-39555861f443">


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
