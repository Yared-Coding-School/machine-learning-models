# Simple Linear Regression
# Load the data 
import pandas as pd


df = pd.read_excel("./class1/data.xlsx")
print(df.head())

# After seeing the data, lets visualize it using graph to make sense of the data
import matplotlib.pyplot as plt

plt.scatter(df['area'], df['price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()

# now lets import the model and initiate it 
from sklearn import linear_model

rg_model = linear_model.LinearRegression()

# separating the features from the target variables 
x = df[["area"]]            # to make it 2d
y = df["price"]

# training the model
rg_model.fit(x, y)

# now we can use the model to pridict the price by giving it a new area value 
y_predict = rg_model.predict([[3300]])
print(y_predict)

# this model is pridicted using linear slop calculations 
# y = mx + b 
# price = slop * area + intercept 
# so lets get thoes numbers 

slop = rg_model.coef_ 
intercept = rg_model.intercept_
print("slop of the line is :", slop.round(2))
print("intercept of the line is :", intercept.round(2))
# now lets create a plot of scatter plot and the line 
# # this is the actual
x = df["area"]
y = df["price"]
plt.scatter(x, y, color = "blue")          

# this is the predicted
y = rg_model.predict(df[["area"]])
plt.plot(x, y, color = "red")       
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()

# now lets pridict using the model for the area_predict file 
area_predict = pd.read_excel("./class1/area_predict.xlsx")
pridicted_prices = rg_model.predict(area_predict)

# creating a new data frame to store the pridicted prices 
area_predict["Pridicted_prices"] = pridicted_prices

# saving the data frame to a new excel file 
area_predict.to_excel("data_with_pridicted_prices.xlsx", index=False)
