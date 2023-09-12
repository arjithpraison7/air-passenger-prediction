import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the local dataset
file_path = "airline-passengers.csv"  # Replace with the actual path
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Display basic statistics
print("Basic Statistics:")
print(df.describe())

# Check the data types of each column
print("\nData Types:")
print(df.dtypes)

# Check the first few rows of the dataset
print("\nFirst Few Rows:")
print(df.head())

# Display information about the dataset
print("\nDataset Information:")
print(df.info())

# Check the range of dates
print("Start Date:", df['Month'].min())
print("End Date:", df['Month'].max())

# Time Series Plot
plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['Passengers'])
plt.title('Monthly International Airline Passengers')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.grid(True)
plt.show()

# Step 2: Data Cleaning

# Check for missing values
missing_values = df.isnull().sum()

# Print out the missing values (if any)
print("Missing Values:")
print(missing_values)

# Step 3: Descriptive Statistics

# Calculate basic summary statistics for the "Passengers" column
passengers_stats = df['Passengers'].describe()

# Print out the statistics
print("Passengers Statistics:")
print(passengers_stats)

# Step 4: Time Series Visualization

# Create a time series plot with rotated x-axis labels
plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['Passengers'])
plt.title('Monthly International Airline Passengers')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.grid(True)
plt.show()  

# Convert 'Month' column to datetime format
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Calculate the rolling mean to visualize the trend
rolling_mean = df['Passengers'].rolling(window=12).mean()

# Calculate the rolling standard deviation to visualize the seasonality
rolling_std = df['Passengers'].rolling(window=12).std()

# Plot the original data, rolling mean, and rolling standard deviation
plt.figure(figsize=(12, 6))
plt.plot(df['Passengers'], label='Original Data')
plt.plot(rolling_mean, label='Rolling Mean (Trend)')
plt.plot(rolling_std, label='Rolling Std (Seasonality)')
plt.title('Original Data, Rolling Mean, and Rolling Std')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Correlation Analysis

# Calculate the correlation between consecutive months
correlation = df['Passengers'].autocorr()

# Print out the correlation coefficient
print(f"Correlation between consecutive months: {correlation}")

# Step 7: Forecasting (Holt-Winters Method)

# Fit the Holt-Winters model
model = ExponentialSmoothing(df['Passengers'], trend='add', seasonal='add', seasonal_periods=12)
result = model.fit()

# Forecast future values
forecast_periods = 12  # Forecast for the next 12 months
forecast = result.forecast(forecast_periods)

# Plot the original data and the forecasted values
plt.figure(figsize=(12, 6))
plt.plot(df['Passengers'], label='Original Data')
plt.plot(forecast, label='Forecasted Values', linestyle='--')
plt.title('Original Data and Forecasted Values')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.legend()
plt.grid(True)
plt.show()
