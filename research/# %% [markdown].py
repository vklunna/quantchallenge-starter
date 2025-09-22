# %% [markdown]
# # Getting Started: Market Research
# This Jupyter notebook is a quick demonstration on how to get started on the market research section.

# %% [markdown]
# ## 1) Download Data
# Please download the train and test data and place it within the ./research/data path. If you've placed it in the correct place, you should see the following cell work:

# %%
import pandas as pd

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

print(train_data.head())
print(test_data.head())
print(test_data.columns)


# %% [markdown]
# ## 2) Investigate the Dataset
# In the datasets, you're given columns of time and A through N, each of which represent some sort of real-life market quantity. In the train dataset, you're also given Y1 and Y2, real-life market quantities you'd like to predict in terms of time and A through N. You're not given Y1 and Y2 in the test set, because this is what you're being asked to predict.
# 
# Let's do some exploration of the relationships of A - N and Y1. In particular, let's look at the relationship between C and Y1:

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(train_data['C'], train_data['Y1'])
plt.xlabel('C')
plt.ylabel('Y1')
plt.title('Relationship between C and Y1')
plt.show()

# %%
# Calculate correlation between C and Y1
correlation = train_data['C'].corr(train_data['Y1'])
print(f"Correlation between C and Y1: {correlation:.4f}")

# %% [markdown]
# Clearly there's a strong relationship between C and Y1. You should definitely use C to predict Y1!

# %%
# check the types of Y1 and Y2
import matplotlib.pyplot as plt
import numpy as np

train_data[["Y1", "Y2"]].dtypes
train_data[["Y1", "Y2"]].nunique()
print(train_data[["Y1","Y2"]].value_counts().head(10))

# %%
%pip install graphviz
# Modelling
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

# %%
from sklearn.model_selection import train_test_split

# Features (drop targets)
X = train_data.drop(columns=["Y1", "Y2", "time"])

# Targets
y1 = train_data["Y1"]
y2 = train_data["Y2"]

# Split 80:20
X_train, X_test, y1_train, y1_test = train_test_split(
    X, y1, test_size=0.2, random_state=42
)
_, _, y2_train, y2_test = train_test_split(
    X, y2, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y1_train shape:", y1_train.shape)
print("y1_test shape:", y1_test.shape)
print("y2_train shape:", y2_train.shape)
print("y2_test shape:", y2_test.shape)


# %%
rf = RandomForestRegressor()
rf.fit(X_train, y1_train)

# %%
yf = RandomForestRegressor()
yf.fit(X_train, y2_train)

# %%
y1_pred = rf.predict(X_test)

y2_pred = yf.predict(X_test)

# %%
from sklearn.metrics import r2_score

r2_y1 = r2_score(y1_test, y1_pred)
r2_y2 = r2_score(y2_test, y2_pred)

print(f"R² Y1: {r2_y1:.4f}")
print(f"R² Y2: {r2_y2:.4f}")
print(f"Average R²: {(r2_y1 + r2_y2)/2:.4f}")


# %% [markdown]
# ## 3) Submit Predictions
# In order to submit predictions, we need to make a CSV file with three columns: id, Y1, and Y2. In the below example, we let our predictions of Y1 and Y2 be the means of Y1 and Y2 in the train set.

# %%
# preds = test_data[['id']]
# preds['Y1'] = train_data['Y1'].mean()
# preds['Y2'] = train_data['Y2'].mean()
# preds

# # %%
# # save preds to csv
# preds.to_csv('preds.csv', index=False)

#%%
#FINE TUNNING
split = int(0.8*len(train_data))
X_train, X_test = train_data.iloc[:split].drop(columns=["Y1","Y2", "time"]), train_data.iloc[split:].drop(columns=["Y1", "Y2"])
y1_train, y1_test = train_data["Y1"].iloc[:split], train_data["Y1"].iloc[split:]
y2_train, y2_test = train_data["Y2"].iloc[:split], train_data["Y2"].iloc[split:]

# use also time series split cross-validation to respect timeline and randomized search cross validation (faster than grid search)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

tss = TimeSeriesSplit(n_splits = 4)
#%%
param = {
    "n_estimators": [200, 300, 400],
    "max_depth": [None, 8, 12, 16],
    "min_samples_split": [10, 20, 40],
    "min_samples_leaf": [4, 8, 16],
    "max_features": ["sqrt", 0.5],
    "bootstrap": [True], 
    "max_samples": [None, 0.7],
    "random_state": [42]
}

rf_y1 = RandomForestRegressor()
search_y1 = RandomizedSearchCV(
    rf_y1, param_distributions=param,
    n_iter=30, scoring="r2", cv=tss, random_state=42
)
search_y1.fit(X_train, y1_train)
print('Best params for Y1:', search_y1.best_params_)

rf_y2 = RandomForestRegressor()
search_y2 = RandomizedSearchCV(
    rf_y2, param_distributions=param,
    n_iter=30, scoring="r2", cv=tss, random_state=42
)
search_y2.fit(X_train, y2_train)
print('Best params for Y2:', search_y2.best_params_)

y1_pred = search_y1.best_estimator_.predict(X_test)
y2_pred = search_y2.best_estimator_.predict(X_test)
r2_y1 = r2_score(y1_test, y1_pred)
r2_y2 = r2_score(y2_test, y2_pred)
print(f'R2 y1: {r2_y1}, R2 y2: {r2_y2}, avg: {(r2_y1+r2_y2)/2}')


# %% [markdown]
# You should now be able to submit preds.csv to [https://quantchallenge.org/dashboard/data/upload-predictions](https://quantchallenge.org/dashboard/data/upload-predictions)! Note that you should receive a public $R^2$ score of $-0.042456$ with this set of predictions. You should try to get the highest possible $R^2$ score over the course of these next few days. Be careful of overfitting to the public score, which is only calculated on a subset of the test data—the final score that counts is the private $R^2$ score!


