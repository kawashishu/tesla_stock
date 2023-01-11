import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data into a pandas DataFrame
df = pd.read_csv("TSLA.csv")

# Convert the Date column to a datetime data type
df["Date"] = pd.to_datetime(df["Date"])

# Handle missing values
df = df.dropna()

# Check for and handle outliers in all columns
for column in df.columns:
    if column == "Date":
        continue
    mean = df[column].mean()
    std = df[column].std()
    df = df[(df[column] > mean - 2*std) & (df[column] < mean + 2*std)]

# Normalize the data
for column in df.columns:
    if column == "Date":
        continue
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

# Split the data into training and testing sets
train_df = df.iloc[:int(len(df)*0.8)]
test_df = df.iloc[int(len(df)*0.8):]

# Calculate the correlations between the columns
corr_matrix = df.corr(numeric_only=False)

# Create a heatmap of the correlations
sns.heatmap(corr_matrix)

# Show the plot
plt.show()

############################

# Split the data into features and targets
X = df.drop(columns=["Date"])
y = df["Close"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Choose a machine learning model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
print("LinearRegression: ", model.score(X_test, y_test))


##################################
# Decision Tree Model
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_rmse = np.sqrt(dt_mse)
print("Decision Tree RMSE: ", dt_rmse)

# Random Forest Model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
print("Random Forest RMSE: ", rf_rmse)

###################################################

# Plot the predicted prices against the actual prices
# Linear Regression Model

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

plt.scatter(y_test, lr_predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Linear Regression Model")
plt.show()


####################################

# Collect new data that contains future dates and any other relevant features
future_data = pd.read_csv("future_data.csv")

# If applicable, preprocess the new data in the same way that you preprocessed the training data
future_data["Date"] = pd.to_datetime(future_data["Date"])
future_data = future_data.dropna()
for column in future_data.columns:
    if column == "Date":
        continue
    future_data[column] = (future_data[column] - future_data[column].min()) / (future_data[column].max() - future_data[column].min())

# Use the predict method on the trained model to make predictions on the new data
future_predictions = model.predict(future_data.drop(columns=["Date"]))

# Use the resulting predictions in a meaningful way
print(future_predictions)
