# Import the libraries

from pyexpat import model
import pyexpat.model as model
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression
import joblib
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Simple preprocessing
df = df[["Sex","Fare","Survived"]].dropna()
df['Sex'] = df["Sex"].map({'male':0,'female':1})

x = df[['Sex','Fare']]
y = df['Survived']

# Train the model and save 
mode = LogisticRegression()
model.fit(x,y)
joblib.dump(model,'titanic_model.pkl')
