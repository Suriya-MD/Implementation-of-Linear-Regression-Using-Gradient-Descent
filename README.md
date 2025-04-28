# Exp-03: Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

Here’s a more concise version of the algorithm in 6 steps:

### Algorithm:
1. **Preprocess Data**: Load the dataset, separate features `x1` and target `y`, and standardize them using `StandardScaler`.
2. **Initialize Parameters**: Add a column of ones to `x1` for the bias term and initialize `theta` as zeros.
3. **Gradient Descent**: For each iteration:
   - Compute predictions: `predictions = X * theta`
   - Calculate errors: `errors = predictions - y`
   - Update `theta`: `theta -= learning_rate * (1/m) * X^T * errors`
4. **Repeat**: Perform the above gradient descent step for `num_iters` iterations.
5. **Prediction**: For a new data point, standardize it, then predict the target using `prediction = new_data * theta`.
6. **Output**: Inverse transform the predicted value and print it.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:     SURIYA M
RegisterNumber:   212223110055
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
    ##coefficient of b
    x=np.c_[np.ones(len(x1)),x1]
    ##initialize theta with zero
    theta=np.zeros(x.shape[1]).reshape(-1,1)
    ##perform gradient decent
    for _ in range(num_iters):
        ##calculate predictions
        predictions=(x).dot(theta).reshape(-1,1)
        ##calculate errors
        errors=(predictions - y).reshape(-1,1)
        ##update theta using gradient descent
        theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv",header=None)
print(data.head())
##assume the last column as your target varible y
x=(data.iloc[1:,:-2].values)
print(x)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
print(x1_scaled)
print(y1_scaled)
##learn model parameters
theta=linear_regression(x1_scaled,y1_scaled)
##predict target value for a new data point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
print("Name: SURIYA M\nReg No: 212223110055")

```

## Output:
### Dataset values
![Screenshot 2025-04-28 141503](https://github.com/user-attachments/assets/0b5b19ca-cf54-4540-8692-d5a982e899f1)

### X Values
![Screenshot 2025-04-28 141641](https://github.com/user-attachments/assets/530210ac-bafe-4216-8907-a5aa361132f3)

![Screenshot 2025-04-28 141736](https://github.com/user-attachments/assets/d321af4a-9317-4a1e-8d31-0d769be389cc)

### Y Values
![Screenshot 2025-04-28 141918](https://github.com/user-attachments/assets/d8aa3da6-a63d-426a-a0a7-f1ec36941199)

![Screenshot 2025-04-28 141943](https://github.com/user-attachments/assets/3f7cc19a-4d9d-4902-babe-61bb37e8ecb5)

### X1_scaled Values
![Screenshot 2025-04-28 142153](https://github.com/user-attachments/assets/71152c63-89f7-4676-b392-8ce960b9d739)

![Screenshot 2025-04-28 142205](https://github.com/user-attachments/assets/db7b0eb1-812a-46cf-ab32-e6afc2771592)

### Y1_scaled Values
![Screenshot 2025-04-28 142217](https://github.com/user-attachments/assets/05c7931a-7e1a-4b6d-950e-0fa311fe58fb)

![Screenshot 2025-04-28 142230](https://github.com/user-attachments/assets/4e3f7c66-71fd-4de9-af0c-11989c72bbbb)

### Predicted Values
![image](https://github.com/user-attachments/assets/54b02916-de39-45df-9d64-8b95854869dc)

## Result:

Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

