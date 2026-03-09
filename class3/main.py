# multivariate models - selecting models by their performance
# model selection 

import joblib
from sklearn import datasets
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
housing = datasets.fetch_california_housing()

#  using polynominal
from sklearn.preprocessing import PolynomialFeatures

x = housing.data
y = housing.target

poly = PolynomialFeatures()
poly.fit_transform(x)

# spliting the data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=100
)

# now lets collect models we want to use 
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# now lets initiate the models
LR = LinearRegression()
GBR = HistGradientBoostingRegressor()
RFR = RandomForestRegressor(
    n_jobs = -1                    # making sure we use all cors of cpu
)

# now we use this models using for loop 

for i in [LR, GBR, RFR]:
    i.fit(x_train, y_train)
    y_predict = i.predict(x_test)
    r2 = r2_score(y_test, y_predict)
    print(f"{i} model result:", r2)


# by running the model we found which model is most effective 
# in our case HistGBR has the best r2 - 0.844

# since we have our model - now we need to do Parameter optimization /Hyperparameterization/
# lets start with 
# max_iter = max number of branches in the tree structure 
# learning_rate = Controls how much each new tree corrects the errors of previous trees.
# Small (0.01): Slow learning, needs more trees, better generalization
# Large (0.3):  Fast learning, fewer trees needed, risk of overshooting
# | Parameter           | What it does                      | Typical range |
# | ------------------- | --------------------------------- | ------------- |
# | `learning_rate`     | Step size for updates             | 0.01 – 0.3    |
# | `max_iter`          | Number of trees (boosting rounds) | 100 – 1000    |
# | `max_depth`         | Tree depth (complexity)           | 3 – 10        |
# | `min_samples_leaf`  | Minimum samples per leaf          | 10 – 50       |
# | `max_leaf_nodes`    | Maximum leaves per tree           | 15 – 50       |
# | `l2_regularization` | Ridge penalty on weights          | 0 – 10        |


for j in [0.1, 0.05, 0.001]:
    for i in [200, 250, 300]:
        model = HistGradientBoostingRegressor(
            max_iter=i,
            learning_rate=j
        )

        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(f"Max Iteration {i},| Learning Rate: {j},| Score: {r2}")


#  this results in 
# Max Iteration 300,| Learning Rate: 0.1,| Score: 0.855796382529971 to be the best

# -------------------------------------------------------------------------------------------

# but if you have multiple params you need to use randomized search cv (cross-validation)
# RandomizedSearchCV — Searches random combinations of hyperparameters instead of all possible combinations.
from sklearn.model_selection import RandomizedSearchCV

# lets create an object of the params 
param_dist = {
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    'max_iter': [100, 200, 250, 300, 500],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_leaf': [5, 10, 20, 50],
    'l2_regularization': [0, 0.01, 0.1, 1, 10]
}

model = HistGradientBoostingRegressor(random_state=100)

# lets initialize the search 
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,                              # Try 50 random combinations
    cv=5,                                   # 5-fold cross-validation
    scoring='r2',                           # Optimize for R² score
    n_jobs=-1,                              # Use all CPU cores
    random_state=100,
    verbose=1                               # Show progress
)

# now lets search for best parameters 
random_search.fit(x_train, y_train)

# printing the best
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Cross Validation Score: {random_search.best_score_:.4f}")

# now lets evalute the best model and prediction ability 

best_model = random_search.best_estimator_

y_predict = best_model.predict(x_test)

# now lets do the r2 score 
r2 = r2_score(y_test, y_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))

print(f"\nTest Set R2: {r2:.4f}")
print(f"Test Set RMSE: {rmse:.2f}")


# now lets save the model in local so that we can use it later 
model = HistGradientBoostingRegressor(
    max_iter=350,
    learning_rate=0.05
)

model.fit(x_train, y_train)

# now lets save the model using joblib
import joblib
joblib.dump(model, "my_best_model.joblib")

# now you will find the model in you root dir
# so lets load it and use it
my_model = joblib.load("my_best_model.joblib")

y_predicted = my_model.predict(x_test)
r_square_score = r2_score(y_test, y_predict)
print("Final r2 value:", r_square_score)