# multiple variable regression - multivariate regression
# importing data from sklearn


from sklearn import datasets
import pandas as pd

housing_datasets = datasets.fetch_california_housing()
# looking at the data
# print(housing_datasets)

# now we can use it as it is, but if you want to change it to a dataframe to see it as a table 
# table = pd.DataFrame(housing_datasets.data, columns=housing_datasets.feature_names)
# table["target"] = housing_datasets.target
# print(table)

# but we can use the data as it is 
x = housing_datasets.data
y = housing_datasets.target

# now since we have our data - we need to split it
from sklearn.model_selection import train_test_split

# TODO: split the data and train the model
x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size=0.2,
    random_state = 432
)

# now lets import the model 
from sklearn.linear_model import LinearRegression

# creating and training the model
model = LinearRegression()
model.fit(x_train, y_train)

# testing the model by pridicting 
y_predict = model.predict(x_test)

# now lets check how accurate the model is in predicting the y_test

from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# now we use r-score to compare the real values with the model prediction
r2 = r2_score(y_test, y_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print("first r2:", r2)                           # ~ 0.608, means the model learns only 61%
print("First RMSE:", rmse)
# this is called NOTE a baseline 

# now we need to optimize the model by using polynomial features 
# Polynomial Features — Create new features by raising existing ones to powers and multiplying them together.
# Why: Capture non-linear relationships that linear models miss.

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()
x = poly.fit_transform(x)

# now lets check the shape
print(x.shape)

# now lets resplit x into train and test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=432)
model = LinearRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print("optimized r2 score:", r2)            # r2 = 0.66 improved by 6 points
print("optimized MSRE score:", rmse)

