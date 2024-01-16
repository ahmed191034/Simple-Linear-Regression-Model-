# Simple-Linear-Regression-Model-
its a simple linear regression model wich could helpus in calculating package of students by just getting cgpa
#  Creating YourModel class
class YourModel:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, x_train, y_train):
        num = 0
        den = 0
        for i in range(len(x_train)):
            num += ((x_train[i] - x_train.mean()) * (y_train[i] - y_train.mean()))
            den += (x_train[i] - x_train.mean()) ** 2
        self.m = num / den
        self.b = y_train.mean() - (self.m * x_train.mean())

    def predict(self, x_test):
        return self.m * x_test + self.b

# Importing necessary libraries
import numpy as np
import pandas as pd

# Read data from CSV
df = pd.read_csv("Placement.csv")

# Selecting the 'cgpa' and 'package' columns
x = df['cgpa'].values
y = df['package'].values

# Importing the train_test_split function
from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Creating an object of our class
Lr = YourModel()

# Fitting the model with training data
Lr.fit(x_train, y_train)

# Making predictions with the model
y_pred = Lr.predict(x_test)
