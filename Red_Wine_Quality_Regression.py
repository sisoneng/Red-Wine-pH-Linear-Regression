import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

np.set_printoptions(precision=2)

## 🎯 Data Collection
def load_file():
    file_path = os.path.join("C:\dev\machinelearning\datasets", "winequality-red.csv")
    return pd.read_csv(file_path)

df_wq = load_file()


# checking for missing values
df_wq.isnull().sum()


## 🎯 Data Analysis and Visualization

correlation = df_wq.corr()


## fixed acidity has the highest correlation on pH level so we used it for stratify sampling
df_wq["fixed acidity"].hist()

## 🎯 Data preprocessing
X = df_wq.drop("pH", axis=1)
Y = df_wq["pH"]

## random sampling
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

df_wq["fixed_acidity_cat"] = pd.cut(df_wq["fixed acidity"], bins=[0, 6, 12, np.inf], labels=[1, 2, 3])

## stratified sampling
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df_wq, df_wq["fixed_acidity_cat"]):
    strat_train = df_wq.loc[train_index]
    strat_test = df_wq.loc[test_index]
    
X_strat_train = strat_train.drop("pH", axis=1)
X_strat_test = strat_test.drop("pH", axis=1)
Y_strat_train = strat_train["pH"]
Y_strat_test = strat_test["pH"]

print("Sampling label shapes: ", "Y shape = ", Y.shape, "Y train shape = ", Y_train.shape, "Y test shape = ", Y_test.shape)

## 🎯 Model Training on Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

## 🎯 Model Evaluation (RMSE and R2)

## RMSE
from sklearn.metrics import mean_squared_error
pH_predictions = lin_reg.predict(X_train)
mse = mean_squared_error(Y_train, pH_predictions)
lr_rmse = np.sqrt(mse)


## k fold cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg, X_train, Y_train,
scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

## predict on training set
predictions = lin_reg.predict(X_train[:5])
print("Prediction on training set")
print("Observed Value", list(Y_train[:5]))
print("Predicted Value: ", predictions)

## predict on test set
predictions = lin_reg.predict(X_test[:5])
print("Prediction on test set")
print("Observed Value: ", list(Y_test[:5]))
print("Predicted Value: ", predictions)

print("RMSE w/o cross-validation = ", round(lr_rmse, 6))
print("RMSE w/ cross-validation = ", round(rmse_scores.mean(), 6))

R2 = lin_reg.score(X_train, Y_train)
print("R2 (training set) =", round(R2 * 100, 2), "%")


R2 = lin_reg.score(X_test, Y_test)
print("R2 (test set) =", round(R2 * 100, 2), "%")

