import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

df = pd.read_csv("salary_regression_dataset.csv")

print(df.head())

x = df.drop('salary', axis=1)
y = df['salary']

# Train test split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# feature scaling->standarization

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

# model training-> linear regression

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

def predictsalary(experience, hours):
    features = np.array([[experience, hours]])
    features_scaled = scalar.transform(features)
    results = lr.predict(features_scaled)
    return results

experience = 4
hoursofwork = 38

predicted_salary = predictsalary(experience, hoursofwork)
print(predicted_salary)

import pickle 

pickle.dump((lr, scalar), open("linear_regression.pkl", "wb"))


