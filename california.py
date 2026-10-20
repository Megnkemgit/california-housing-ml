
# ============================================================
# 1. IMPORT LIBRARIES AND LOAD THE DATA
# ============================================================

# Load the California housing dataset
from sklearn.datasets import fetch_california_housing

# Import pandas to work with data in tables
import pandas as pd

# Import matplotlib and seaborn to create graphs
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset as a pandas DataFrame
housing = fetch_california_housing(as_frame=True)

# Store the housing table in a variable called df
df = housing.frame

# Display the first 5 rows
print(df.head())


# ============================================================
# 2. EXPLORE THE DATA
# ============================================================

# Show the column names, data types, and missing values
df.info()

# Show summary statistics, such as average, minimum, and maximum
print(df.describe())


# ============================================================
# 3. CREATE HISTOGRAMS
# ============================================================

# Create a histogram for every numerical column
# figsize controls the overall size of the graphs
df.hist(figsize=(12, 8))

# Display the histograms
plt.tight_layout()
plt.show()


# ============================================================
# 4. CREATE A CORRELATION HEATMAP
# ============================================================

# Correlation measures how strongly two numerical variables
# move together. Values range from -1 to +1.
#
# +1 = strong positive relationship
#  0 = little or no linear relationship
# -1 = strong negative relationship

plt.figure(figsize=(10, 8))

# Calculate correlations and display them as a colored table
# annot=True displays the correlation numbers
# cmap="coolwarm" uses colors to distinguish positive and negative values
sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Between Housing Variables")
plt.tight_layout()
plt.show()


# ============================================================
# 5. SPLIT THE DATA INTO TRAINING AND TEST SETS
# ============================================================

from sklearn.model_selection import train_test_split

# Split the data into two groups:
# 80% for training the model
# 20% for testing how well the model works on unseen data
#
# random_state=42 makes the split reproducible
train_set, test_set = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

# Display the sizes of both groups
print("Training set shape:", train_set.shape)
print("Test set shape:", test_set.shape)


# ============================================================
# 6. SEPARATE FEATURES FROM THE TARGET
# ============================================================

# Features (X) are the information used to predict house values.
# We remove MedHouseVal because that is the value we want to predict.
X_train = train_set.drop("MedHouseVal", axis=1)
X_test = test_set.drop("MedHouseVal", axis=1)

# Target (y) is the answer the model is trying to predict.
y_train = train_set["MedHouseVal"]
y_test = test_set["MedHouseVal"]


# ============================================================
# 7. STANDARDIZE THE FEATURES
# ============================================================

from sklearn.preprocessing import StandardScaler

# StandardScaler adjusts numerical features to have approximately:
# mean = 0
# standard deviation = 1
scaler = StandardScaler()

# Learn the scaling values from the training data and apply them
X_train_scaled = scaler.fit_transform(X_train)

# Apply the SAME scaling values to the test data.
# Do not fit a new scaler on the test data.
X_test_scaled = scaler.transform(X_test)

# Convert the scaled training data back into a readable table
scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

# Check whether the features were standardized correctly
print("Feature means:")
print(scaled_df.mean().round(2))

print("\nFeature standard deviations:")
print(scaled_df.std().round(2))


# ============================================================
# 8. TRAIN A LINEAR REGRESSION MODEL
# ============================================================

from sklearn.linear_model import LinearRegression

# Create a Linear Regression model
model = LinearRegression()

# Train the model using the scaled training features
# and the known training house values
model.fit(X_train_scaled, y_train)


# ============================================================
# 9. EVALUATE THE LINEAR REGRESSION MODEL
# ============================================================

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Ask the model to predict house values for unseen test data
predictions = model.predict(X_test_scaled)

# RMSE measures the typical prediction error,
# with larger errors penalized more heavily.
rmse = mean_squared_error(y_test, predictions) ** 0.5

# MAE measures the average absolute difference
# between the predicted and actual house values.
mae = mean_absolute_error(y_test, predictions)

print("\nLinear Regression Test Results")
print("RMSE:", round(rmse, 3))
print("MAE:", round(mae, 3))


# ============================================================
# 10. USE CROSS-VALIDATION
# ============================================================

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# A pipeline combines scaling and model training.
# During cross-validation, the scaler is fitted separately
# within each training fold to help prevent data leakage.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# Evaluate the pipeline using 5-fold cross-validation.
# The data is divided into 5 parts; each part is used
# for validation once while the others are used for training.
#
# The scoring method returns negative RMSE, so we change
# the scores back to positive numbers for easier reading.
scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    scoring="neg_root_mean_squared_error",
    cv=5
)

print("\nCross-validation RMSE scores:")
print(-scores.round(3))

print("Average cross-validation RMSE:", round((-scores).mean(), 3))


# ============================================================
# 11. SEARCH FOR THE BEST RANDOM FOREST SETTINGS
#     USING GRID SEARCH
# ============================================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# A Random Forest combines many decision trees to make predictions.
#
# n_estimators = number of trees
# max_depth = maximum depth of each tree
#
# GridSearchCV tests every combination listed below.
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None]
}

# Create a search that tests each combination using 3-fold
# cross-validation. Lower RMSE means better predictions.
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

# Train and compare the different parameter combinations
grid_search.fit(X_train, y_train)

print("\nGrid Search Best Parameters:")
print(grid_search.best_params_)

print("Grid Search Best Cross-Validation RMSE:")
print(round(-grid_search.best_score_, 3))


# ============================================================
# 12. SEARCH FOR SETTINGS USING RANDOMIZED SEARCH
# ============================================================

from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

# Instead of testing every possible combination,
# RandomizedSearchCV samples a selection of combinations.
#
# randint(50, 200) samples integers from 50 to 199.
# randint(3, 20) samples integers from 3 to 19.
param_dist = {
    "n_estimators": randint(50, 200),
    "max_depth": randint(3, 20)
}

# Try 20 randomly selected combinations.
# Each combination is evaluated with 3-fold cross-validation.
random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1
)

# Find the best combination among the sampled settings
random_search.fit(X_train, y_train)

print("\nRandomized Search Best Parameters:")
print(random_search.best_params_)

print("Randomized Search Best Cross-Validation RMSE:")
print(round(-random_search.best_score_, 3))


# ============================================================
# 13. TEST THE BEST RANDOM FOREST MODEL
# ============================================================

# Select the best model found by grid search
best_model = grid_search.best_estimator_

# Make predictions on the test set, which was not used
# during model training or parameter selection
rf_predictions = best_model.predict(X_test)

# Calculate the final test errors
rf_rmse = mean_squared_error(y_test, rf_predictions) ** 0.5
rf_mae = mean_absolute_error(y_test, rf_predictions)

print("\nBest Random Forest Test Results")
print("Test RMSE:", round(rf_rmse, 3))
print("Test MAE:", round(rf_mae, 3))