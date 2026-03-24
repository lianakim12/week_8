# %%
# Q1: Data preparation
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
# %%
df = pd.read_csv("https://raw.githubusercontent.com/DS3001/linearRegression/refs/heads/main/data/Q1_clean.csv")

# Rename column for ease
df = df.rename(columns={"Neighbourhood ": 'Neighborhood'})
df.head()
# %%
# Q1: Part 1
# Compute the average prices and scores by `Neighborhood `; which borough is 
# the most expensive on average? Create a kernel density plot of price and log 
# price, grouping by `Neighborhood `.

df_grouped = df.groupby("Neighborhood")[["Price","Review Scores Rating"]].mean()
top_neighborhood = df_grouped.sort_values(by="Price", ascending=False).index[0]

print(f"Average prices and scores by neighborhood: {df_grouped}")
print(f"Most expensive neighborhood: {top_neighborhood}")

# Kernel density plots grouped by neighborhood

sns.kdeplot(data=df, x="Price", hue="Neighborhood")
plt.title("Kernel Density Plot of Price by Neighborhood")
plt.show()

df['Log Price'] = np.log(df['Price'] + 1)  # add 1 to avoid log(0)
sns.kdeplot(data=df, x="Log Price", hue="Neighborhood")
plt.title("Kernel Density Plot of Log Price by Neighborhood")
plt.show()
# %%
# Q1: Part 2
# Regress price on `Neighborhood ` by creating the appropriate dummy/one-hot-encoded
# variables, without an intercept in the linear model. Compare the coefficients in
# the regression to the table from part 1. What pattern do you see? What are the 
# coefficients in a regression of a continuous variable on one categorical variable?

df_without = pd.get_dummies(data=df, columns=["Neighborhood"], drop_first=False,
                    prefix="Nbh") # drop_first=False does not remove the base level)
df_without.head()
# %%
X = df_without[['Nbh_Bronx', 'Nbh_Brooklyn', 'Nbh_Manhattan', 'Nbh_Queens', 'Nbh_Staten Island']]
Y = df_without['Price']

# Without an intercept
model_without = LinearRegression(fit_intercept=False).fit(X, Y)
print(f"Without Intercept: Coefficient = {model_without.coef_}, Intercept = {model_without.intercept_:.4f}")

# Bronx has a coefficient of 75.28. Average price was 75.28. They match
# Brooklyn has a coefficient of 127.75. Average price was 127.75. They match.
# Manhattan has a coefficient of 183.66. Average price was 183.66. They match.
# Queens has a coefficient of 96.86. Average price was 96.86. They match.
# Staten Island has a coefficient of 146.17. Average price was 146.17. They match.

# We can conclude that the coefficients in a regression of a continuous variable on 
# one categorical variable is the same as the average of the variable for each of
# the levels in the categorical variable.
# %%
# Q1: Part 3
# Repeat part 2, but leave an intercept in the linear model. How do you have to handle
# the creation of the dummies differently? What is the intercept? Interpret the 
# coefficients. How can I get the coefficients in part 2 from these new coefficients?

df_with = pd.get_dummies(data=df, columns=["Neighborhood"], drop_first=True,
                    prefix="Nbh") # drop_first=True removes the base level)
df_with.head()

# update X to account for the dropped base level
X = df_with[['Nbh_Brooklyn', 'Nbh_Manhattan', 'Nbh_Queens', 'Nbh_Staten Island']]

model_with = LinearRegression(fit_intercept=True).fit(X, Y)
print(f"With Intercept: Coefficient = {model_with.coef_}, Intercept = {model_with.intercept_:.4f}")

# We have to handle the creation of dummies differently (i.e. drop base level) since
# having the intercept accounts for the base level.

# The intercept 75.28 is the average price for Bronx. It is equal to the Bronx 
# coefficient from part 2, which was also 75.28.

# Given that the neighborhood is Brooklyn, the average price is 52.47 higher than Bronx.
# Given that the neighborhood is Manhattan, the average price is 108.39 higher than Bronx.
# Given that the neighborhood is Queens, the average price is 21.58 higher than Bronx.
# Given that the neighborhood is Staten Island, the average price is 70.89 higher than Bronx.

# Adding the intercept to the new coefficients gets us the coefficients from part 2.
# For Bronx, the intercept is the same as the coefficient from part 2.
# %%
# Q1: Part 4
# Split the sample 80/20 into a training and a test set. Run a regression of `Price` 
# on `Review Scores Rating` and `Neighborhood `. What is the $R^2$ and RMSE on the 
# test set? What is the coefficient on `Review Scores Rating`? What is the most 
# expensive kind of property you can rent?

train, test = train_test_split(df_with, test_size=0.2, random_state=42)

X_train = train[['Review Scores Rating', 'Nbh_Brooklyn', 'Nbh_Manhattan', 'Nbh_Queens', 'Nbh_Staten Island']]
Y_train = train['Price']

X_test = test[['Review Scores Rating', 'Nbh_Brooklyn', 'Nbh_Manhattan', 'Nbh_Queens', 'Nbh_Staten Island']]
Y_test = test['Price']

model_new = LinearRegression(fit_intercept=True).fit(X_train, Y_train)
Y_pred = model_new.predict(X_test)

r2 = r2_score(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print(f"R^2: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"All Coefficients = {model_new.coef_}")
print(f"Coefficient on Review Scores Rating = {model_new.coef_[0]:.4f}")

# Manhattan has the highest predicted value, so it is the most expensive.
# We determine this by comparing the predicted prices for each neighborhood, 
# which we can find by adding the intercept and the corresponding coefficient. 
# %%
# Q1: Part 5
# Run a regression of `Price` on `Review Scores Rating` and `Neighborhood ` and 
# `Property Type`. What is the $R^2$ and RMSE on the test set? What is the coefficient
# on `Review Scores Rating`? What is the most expensive kind of property you can rent?

df_updated = pd.get_dummies(data=df_with, columns=["Property Type"], drop_first=True,
                    prefix="Ty") # drop_first=True removes the base level)
df_updated.head()
df_updated.columns
# %%
# Specify all variables that we will use for part 5
df_updated = df_updated[['Price', 'Review Scores Rating', 'Nbh_Brooklyn', 
                         'Nbh_Manhattan', 'Nbh_Queens', 'Nbh_Staten Island', 
                         'Ty_Bed & Breakfast', 'Ty_Boat', 'Ty_Bungalow', 'Ty_Cabin', 
                         'Ty_Camper/RV', 'Ty_Castle', 'Ty_Chalet', 'Ty_Condominium',
                         'Ty_Dorm', 'Ty_House', 'Ty_Hut', 'Ty_Lighthouse', 'Ty_Loft', 
                         'Ty_Other','Ty_Townhouse', 'Ty_Treehouse', 'Ty_Villa']]

train, test = train_test_split(df_updated, test_size=0.2, random_state=42)

X_train = train.drop(columns=["Price"])
Y_train = train["Price"]

X_test = test.drop(columns=["Price"])
Y_test = test["Price"]

model_new_ = LinearRegression(fit_intercept=True).fit(X_train, Y_train)
Y_pred = model_new_.predict(X_test)

r2 = r2_score(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print(f"R^2: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"All Coefficients = {model_new_.coef_}")
print(f"Coefficient on Review Scores Rating = {model_new_.coef_[0]:.4f}")

# Bungalow has the highest predicted value, so it is the most expensive.
# We determine this by comparing the predicted prices for each property type, 
# which we can find by adding the intercept and the corresponding coefficient. 
# %%
# Q1: Part 6
# What does the coefficient on `Review Scores Rating` mean if it changes from part 4 
# to 5? Hint: Think about how multiple linear regression works.

# Coefficient on Review Scores Rating from part 4: 1.2119
# Coefficient on Review Scores Rating from part 5: 1.2010
# They are not very different

# The coefficient on Review Scores Rating represents the expected change in price for 
# a one-unit increase in review score, given that all other variables are constant.
# Adding Property Type to the model barely affects the coefficient on RSR.
# This means that RSR is not strongly related to Property Type, and therefore its 
# effect on price is about the same across different property types.
# %%
# Q2: Part 1
# Load `cars_hw.csv`. These data were really dirty, and I've already cleaned them a 
# significant amount in terms of missing values and other issues, but some issues 
# remain (e.g. outliers, badly skewed variables that require a log or arcsinh 
# transformation) Note this is different than normalizing: there is a text below that 
# explains further. Clean the data however you think is most appropriate.

df = pd.read_csv("/workspaces/week_8/cars_hw.csv")
df = df.iloc[:, 1:] # remove the index column
df.head()
# %%
# Check distribution of Y variable
plt.hist(df['Price'])

# Histogram shows a right skew; we should not use this
# %%
# Attempt transforming the Y variable
df['Log Price'] = np.log(df['Price'] + 1)  # add 1 to avoid log(0)
plt.hist(df['Log Price'])

# Histogram is much more normally distributed; we should use this
# %%
# Check for outliers in log Y variable
Q1 = df['Log Price'].quantile(0.25)
Q3 = df['Log Price'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['Log Price'] < lower) | (df['Log Price'] > upper)]

print(f"Number of outliers: {len(outliers)}")
# %%
# Drop the outliers we got
df = df.drop(index=outliers.index)
# %%
# Q2: Part 2
# Summarize the `Price` variable and create a kernel density plot. Use `.groupby()` 
# and `.describe()` to summarize prices by brand (`Make`). Make a grouped kernel 
# density plot by `Make`. Which car brands are the most expensive? What do prices 
# look like in general?

summary = df.groupby('Make')['Price'].describe()
summary
# %%
sns.kdeplot(data=df, x="Price", hue="Make")
plt.title("Kernel Density Plot of Price by Make")
plt.show()

# The most expensive car brands are MG Motors, Kia, and Jeep. They have the highest
# average prices, and the KDE plot also seems to support this observation.
# They are also the only Makes that have a minimum price of 1 million dollars.
# %%
# Q3: Part 3
# Split the data into an 80% training set and a 20% testing set.

train, test = train_test_split(df, test_size=0.2, random_state=42)

X_train = train.drop(columns=["Price", "Log Price"])
Y_train = train["Log Price"]

X_test = test.drop(columns=["Price", "Log Price"])
Y_test = test["Log Price"]
# %%
# Q3: Part 4
# Make a model where you regress price on the numeric variables alone; what is the 
# $R^2$ and `RMSE` on the training set and test set? Make a second model where, for 
# the categorical variables, you regress price on a model comprised of one-hot encoded 
# regressors/features alone (you can use `pd.get_dummies()`; be careful of the dummy 
# variable trap); what is the $R^2$ and `RMSE` on the test set? Which model performs 
# better on the test set? Make a third model that combines all the regressors from the 
# previous two; what is the $R^2$ and `RMSE` on the test set? Does the joint model 
# perform better or worse, and by home much?

# First model: only numeric variables

numeric_cols = ['Make_Year', 'Mileage_Run', 'Seating_Capacity']
X_train_num = X_train[numeric_cols]
X_test_num = X_test[numeric_cols]

model_num = LinearRegression(fit_intercept=True).fit(X_train_num, Y_train)
Y_pred_num = model_num.predict(X_test_num)

r2 = r2_score(Y_test, Y_pred_num)
mse = mean_squared_error(Y_test, Y_pred_num)
rmse = np.sqrt(mse)

print(f"R^2: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
# %%
# Second model: only categorical variables

categorical_cols = ['Make', 'Color', 'Body_Type', 'No_of_Owners', 'Fuel_Type', 
                    'Transmission', 'Transmission_Type']
prefixes = ['M', 'C', 'BT', 'NO', 'FT', 'T', 'TT']
df_cat = pd.get_dummies(data=df, columns=categorical_cols, drop_first=True,
                    prefix=prefixes) # drop_first=True removes the base level)

train, test = train_test_split(df_cat, test_size=0.2, random_state=42)

X_train_cat = train.drop(columns=numeric_cols)
X_train_cat = X_train_cat.drop(columns=["Price", "Log Price"])

X_test_cat = test.drop(columns=numeric_cols)
X_test_cat = X_test_cat.drop(columns=["Price", "Log Price"])

model_cat = LinearRegression(fit_intercept=True).fit(X_train_cat, Y_train)
Y_pred_cat = model_cat.predict(X_test_cat)

r2 = r2_score(Y_test, Y_pred_cat)
mse = mean_squared_error(Y_test, Y_pred_cat)
rmse = np.sqrt(mse)

print(f"R^2: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# The second model performed better because it had a higher R^2 and a lower RMSE.
# %%
# Third model: all variables

X_train_all = train.drop(columns=["Price", "Log Price"])
X_test_all = test.drop(columns=["Price", "Log Price"])

model_all = LinearRegression(fit_intercept=True).fit(X_train_all, Y_train)
Y_pred_all = model_all.predict(X_test_all)

r2 = r2_score(Y_test, Y_pred_all)
mse = mean_squared_error(Y_test, Y_pred_all)
rmse = np.sqrt(mse)

print(f"R^2: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# This model performed the best because it had the highest R^2 and the lowest RMSE.
# %%
# Q2: Part 5
# Use the `PolynomialFeatures` function from `sklearn` to expand the set of numerical 
# variables you're using in the regression. As you increase the degree of the 
# expansion, how do the $R^2$ and `RMSE` change? At what point does $R^2$ go negative 
# on the test set? For your best model with expanded features, what is the $R^2$ and 
# `RMSE`? How does it compare to your best model from part 4?

results = {}
for degree in range(1, 11):
    pf  = PolynomialFeatures(degree=degree, include_bias=False)
    Xtr = pf.fit_transform(X_train_num)
    Xte = pf.transform(X_test_num)
    m = LinearRegression().fit(Xtr, Y_train)
    Y_pred = m.predict(Xte)
    r2 = r2_score(Y_test, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    results[f'degree_{degree}'] = {"R2": r2, "RMSE": rmse}
    print(f"Degree {degree} -  R²: {r2:.4f},  RMSE: {rmse:.4f}")

# As the degree of expansion increases, R^2 increases then decreases. R^2 goes negative
# at degree 4, where it decreases from 0.4305 to 0.4137. RMSE keeps increasing.

# Our best model is at degree 3, where R^2 is 0.4305 and RMSE is 0.3277.

# This model does not perform as well as the best model from part 4.
# %%
# Q2: Part 6
# For your best model so far, determine the predicted values for the test data and plot
# them against the true values. Do the predicted values and true values roughly line
# up along the diagonal, or not? Compute the residuals/errors for the test data and 
# create a kernel density plot. Do the residuals look roughly bell-shaped around zero?
# Evaluate the strengths and weaknesses of your model.

plt.figure(figsize=(6, 6))
plt.scatter(Y_test, Y_pred_all, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted Values')
plt.show()

# Yes, the predicted values and true values do roughly line up along the diagonal.
# %%
residuals = Y_test - Y_pred_all

sns.kdeplot(residuals)
plt.title("Kernel Density Plot of Residuals")
plt.xlabel("Residuals")
plt.show()

# Yes, the residuals look roughly bell-shaped around zero.
# %%
r2 = r2_score(Y_test, Y_pred_all)
mse = mean_squared_error(Y_test, Y_pred_all)
rmse = np.sqrt(mse)

print(f"R^2: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Strengths:
# R^2 is decently high and RMSE is decently low. Both are good signs.
# This model accounts for both numeric and categorical variables in the data.
# The scatterplot shows that the predicted values line up fairly well with true values.
# The KDE plot shows that residuals are roughly centered around zero, which is good.

# Weaknesses:
# This model only captures linear relationships and does not address degrees of expansion.
# This model includes all variables, even ones that may not contribute to good predictions.
# The scatterplot shows that there are some values that were not well predicted.
# The KDE plot shows that there is a slight tail on the left, which is not ideal.
# %%
# Q3: Part 1
# Find a dataset on a topic you're interested in. Some easy options are data.gov, 
# kaggle.com, and data.world.

import kagglehub
import os

path = kagglehub.dataset_download("nikhil7280/student-performance-multiple-linear-regression")
df = pd.read_csv(os.path.join(path, "Student_Performance.csv"))
df.head()
# %%
# Q3: Part 2
# Clean the data and do some exploratory data analysis on key variables that interest 
# you. Pick a particular target/outcome variable and features/predictors.

# Check distribution of Y variable
plt.hist(df['Performance Index'])
# It looks roughly normally distributed, so it is good to go
# %%
# Convert the categorical variable into dummy variables
df_cat = pd.get_dummies(data=df, columns=["Extracurricular Activities"], drop_first=False,
                    prefix="EA")
df_cat.head()
# %%
# Q3: Part 3
# Split the sample into an ~80% training set and a ~20% test set.

train, test = train_test_split(df_cat, test_size=0.2, random_state=42)

X_train = train.drop(columns=["Performance Index"])
Y_train = train["Performance Index"]

X_test = test.drop(columns=["Performance Index"])
Y_test = test["Performance Index"]
# %%
# Q3: Part 4
# Run a few regressions of your target/outcome variable on a variety of features/
# predictors. Compute the RMSE on the test set.

# %%
# Q3: Part 5
# Which model performed the best, and why?

# %%
# Q3: Part 6
# What did you learn?