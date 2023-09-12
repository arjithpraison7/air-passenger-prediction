# Description

In this project, we will perform an in-depth Exploratory Data Analysis (EDA) on the "Air Passengers" dataset. This dataset contains monthly data on the number of international airline passengers from 1949 to 1960.

**Objectives:**

1. **Data Understanding**: Begin by loading and inspecting the dataset. Understand the structure of the data and the information it contains.
2. **Data Cleaning**: Address any missing values or anomalies in the dataset. Ensure the data is ready for analysis.
3. **Descriptive Statistics**: Calculate basic statistics to get an overview of the data's central tendencies and distributions.
4. **Time Series Visualization**: Create time series plots to observe trends, seasonality, and potential patterns in the data.
5. **Seasonal Decomposition**: Use techniques like seasonal decomposition of time series (STL) to break down the series into its components (trend, seasonal, and residual).
6. **Correlation Analysis**: Explore relationships between time periods, and check if there are any noticeable patterns or dependencies.
7. **Forecasting (Optional)**: Building a simple forecasting model to predict future passenger counts.
8. **Conclusions and Insights**: Summarising our findings. What are the main trends, seasonality patterns, and any other insights we’ve discovered?

**Deliverables:**

1. **Jupyter Notebook**: Document our analysis in a Jupyter notebook, including code, visualisations, and explanations.
2. **Visual Reports**: Include visualizations like time series plots, decomposition plots, and any other relevant graphs.
3. **Summary and Conclusions**: Write a concise summary of your findings, discussing the key insights from your analysis.
4. **Forecasting Model** : Forecasting  model and its performance metrics.

## Libraries Used

- pandas
- matplotlib
- seaborn
- statsmodels

## Toolkit Used

- Jupyter Notebook

## Problem Statement

The dataset contains monthly records of international airline passengers from 1949 to 1960. Understanding the historical trends in airline passenger traffic is crucial for optimizing resources, scheduling, and making informed business decisions.

The goal of this project is to perform an in-depth Exploratory Data Analysis (EDA) on the Air Passengers dataset. This analysis will provide valuable insights into the historical trends, seasonality, and potential patterns in airline passenger traffic.

## Steps

**Step 1: Data Understanding and Loading**

- Load the Air Passengers dataset.
- Understand the structure of the data (number of observations, variables, etc.).
- Check the first few rows to get a sense of the data.

**Step 2: Data Cleaning**

- Check for missing values and handle them appropriately.
- Address any anomalies or inconsistencies in the data.

**Step 3: Descriptive Statistics**

- Calculate basic summary statistics (mean, median, standard deviation, etc.) for the passenger counts.

**Step 4: Time Series Visualization**

- Create time series plots to visualize the trend in passenger traffic over time.
- Observe any patterns, trends, or seasonality in the data.

**Step 5: Seasonal Decomposition**

- Use techniques like seasonal decomposition of time series (STL) to break down the series into its components (trend, seasonal, and residual).

**Step 6: Correlation Analysis**

- Explore relationships between different time periods.
- Check for any noticeable patterns or dependencies.

**Step 7:  Forecasting** 

- For forecasting, develop a simple forecasting model (e.g., Holt-Winters) to predict future passenger counts.

**Step 8: Conclusions and Insights**

- Summarising our findings, including the main trends, seasonality patterns, and any other insights we’ve discovered.

## Data Understanding and Loading

To get started, we'll first need to load the "Air Passengers" dataset. The dataset can be acquired in the following link: [Air Passengers Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv). This is loaded using the following lines of code:

```python
import pandas as pd

# Load the local dataset
file_path = "airline-passengers.csv"  # Replace with the actual path
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())
```

The dataset is pretty much straight forward with only two columns and the metadata about these two columns are given bellow:

**Dataset Overview:**

- The dataset contains monthly records of international airline passengers from 1949 to 1960.

**Column Descriptions:**

1. **Month**: This column represents the time period. It's likely in a format like "yyyy-mm" or similar.
2. **Passengers**: This column represents the number of international airline passengers for each month.

### Initial Data Exploration

Initial Data Exploration is carried out using the following lines of code:

```python
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
```

We get the following output

```python
Basic Statistics:
       Passengers
count  144.000000
mean   280.298611
std    119.966317
min    104.000000
25%    180.000000
50%    265.500000
75%    360.500000
max    622.000000

Data Types:
Month         object
Passengers     int64
dtype: object

First Few Rows:
     Month  Passengers
0  1949-01         112
1  1949-02         118
2  1949-03         132
3  1949-04         129
4  1949-05         121

Dataset Information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 144 entries, 0 to 143
Data columns (total 2 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   Month       144 non-null    object
 1   Passengers  144 non-null    int64
```

**Interpretation:**

- The **`describe()`** function provides basic statistics like mean, standard deviation, minimum, maximum, etc., for the "Passengers" column. This gives us an overview of the distribution of passenger counts.
- The **`dtypes`** attribute tells us the data types of each column. Make sure the "Month" column is in a date format and "Passengers" is in an appropriate numerical format.
- The **`head()`** function displays the first few rows of the dataset. This gives us a visual sense of how the data is structured.
- The **`info()`** method provides a summary of the dataset, including the number of non-null entries and the data types of each column.

**Time Period Coverage:**

- It's important to know the range of dates covered in the dataset.

```python
# Check the range of dates
print("Start Date:", df['Month'].min())
print("End Date:", df['Month'].max())
```

**Visualization:**

- Visualizations can provide further insights into the data.

```python
import matplotlib.pyplot as plt

# Time Series Plot
plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['Passengers'])
plt.title('Monthly International Airline Passengers')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.grid(True)
plt.show()
```

![monthly-passengers.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/646c8c4b-8d3a-4906-8820-4bc89ac013b1/264fefcc-4269-4511-9b31-05d5512aca4a/monthly-passengers.png)

## Data Cleaning

Since this dataset is commonly used for educational purposes, it typically doesn't have missing values or anomalies. Nevertheless, it's always good practice to perform this check.

Here's the code to check for missing values and perform any necessary data cleaning:

```python
# Check for missing values
missing_values = df.isnull().sum()

# Print out the missing values (if any)
print("Missing Values:")
print(missing_values)
```

As expected we get the following output:

```python
Missing Values:
Month         0
Passengers    0
```

Since we do not have any missing values we can proceed further about with the Exploratory Data Analysis

## Descriptive Statistics

In this step, we'll calculate basic summary statistics for the "Passengers" column. This will give us an overview of the distribution of passenger counts.

Here's the code to perform this step:

```python
# Step 3: Descriptive Statistics

# Calculate basic summary statistics for the "Passengers" column
passengers_stats = df['Passengers'].describe()

# Print out the statistics
print("Passengers Statistics:")
print(passengers_stats)
```

## Time Series Visualisation

In this step, we'll create a time series plot to visualize the trend in passenger traffic over time. This will help us observe any patterns, trends, or seasonality in the data.

Here's the code to generate the time series plot:

```python
import matplotlib.pyplot as plt

# Step 4: Time Series Visualization

# Create a time series plot
plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['Passengers'])
plt.title('Monthly International Airline Passengers')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.grid(True)
plt.show()
```

![monthly-passengers-refined.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/646c8c4b-8d3a-4906-8820-4bc89ac013b1/19080152-bfb5-441c-84f9-33279fcef787/monthly-passengers-refined.png)

As seen from the plot above we can see that every year there is a peak during summer time. The seasonal jumps around the summer months likely indicate a recurring pattern or seasonality in the data. This is a common characteristic of time series data, especially in areas like travel, where there are often peaks during holiday seasons.

Since this dataset covers international airline passengers, it's not surprising to see these spikes during the summer months, which are typically associated with increased travel.

In our further analysis, we want to dive deeper into this seasonality. We can use techniques like seasonal decomposition of time series (STL) to break down the series into its components (trend, seasonal, and residual). This will allow us to isolate and understand the seasonal patterns more effectively.

Let's move on to **Step 5: Seasonal Decomposition**.

In this step, we'll use the seasonal decomposition of time series (STL) method to break down the series into its components: trend, seasonal, and residual.

Here's the code to perform the seasonal decomposition analysis:

```python
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
```

This code calculates and visualizes the rolling mean (which represents the trend) and the rolling standard deviation (which can indicate seasonality) of the data. This will give us insights into the underlying patterns.

![time-series-visualisation.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/646c8c4b-8d3a-4906-8820-4bc89ac013b1/9984a377-452d-4488-96b5-c60eff18531a/time-series-visualisation.png)

The plot shows three components:

1. **Original Data (Passengers):** This is the actual number of international airline passengers for each month. It's the raw data we loaded from the dataset.
2. **Rolling Mean (Trend):** This is a smoothed version of the data. It's calculated by taking the average of the data over a specified window. In this case, we're using a window of 12 months (1 year). The rolling mean provides an indication of the overall trend in the data.
3. **Rolling Standard Deviation (Seasonality):** This measures the variability or fluctuations in the data over time. Similar to the rolling mean, it's calculated over a window of 12 months. When you observe peaks in the rolling standard deviation, it suggests the presence of seasonality.

**Interpretation:**

- **Trend:**
    - The rolling mean (trend) shows an increasing pattern over time, which suggests a general upward trend in the number of airline passengers. This indicates that, on average, the number of passengers is increasing over the years.
- **Seasonality:**
    - The rolling standard deviation shows periodic fluctuations, indicating the presence of seasonality. Specifically, you can observe regular peaks and valleys in the standard deviation. These peaks occur around the same months each year, suggesting a seasonal pattern.

**Conclusion:**
Based on this analysis, it appears that the data exhibits both an increasing trend and seasonal patterns. This is typical in time series data, especially in domains like travel where there are seasonal variations in demand.

This visual analysis provides valuable insights into the underlying patterns in the data. Further analysis, such as time series decomposition or modeling, can be performed to quantify and forecast these patterns.

## Correlation Analysis

In this step, we'll explore relationships between different time periods. We'll check if there are any noticeable patterns or dependencies.

```python
# Step 6: Correlation Analysis

# Calculate the correlation between consecutive months
correlation = df['Passengers'].autocorr()

# Print out the correlation coefficient
print(f"Correlation between consecutive months: {correlation}")
```

In this code, we're using the **`autocorr()`** function to calculate the autocorrelation, which measures the correlation between consecutive months. The result will be a correlation coefficient indicating the strength and direction of the relationship.

The correlation coefficient ranges from -1 to 1.

- If it's close to 1, it indicates a strong positive correlation (as one variable increases, the other tends to increase as well).
- If it's close to -1, it indicates a strong negative correlation (as one variable increases, the other tends to decrease).
- If it's close to 0, it indicates a weak or no linear correlation between the variables.

Running this code will give us the correlation coefficient between consecutive months. This helps you understand how each month's passenger count relates to the previous month's count.

```python
Correlation between consecutive months: 0.9601946480498522
```

A value of approximately 0.96 indicates a very strong positive correlation between consecutive months. This means that there's a high degree of similarity or relationship between the number of passengers in one month and the number of passengers in the next month.

This finding aligns with our visual observation of an increasing trend in the data, which suggests a consistent upward pattern over time.

Given this high correlation, it further supports the presence of a strong underlying trend in the data.

## Forecasting

Forecasting involves making predictions about future values based on historical data. Let's proceed with building a simple forecasting model using the Holt-Winters method for exponential smoothing.

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Step 7: Optional - Forecasting (Holt-Winters Method)

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
```

![future-prediction.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/646c8c4b-8d3a-4906-8820-4bc89ac013b1/31981af0-ce0c-432a-9cad-ed15d7ecbb71/future-prediction.png)

The dotted red lines show the predicted future forecasted values for the number of passengers travelling in the next year.

n this code, we're using the Holt-Winters method for exponential smoothing. This method is suitable for time series data with trends and seasonality.

Here's a brief explanation of the code:

- We're fitting an Exponential Smoothing model to the data using the **`ExponentialSmoothing`** class from **`statsmodels`**.
- The parameters **`trend='add'`** and **`seasonal='add'`** indicate that we're considering both additive trend and additive seasonal components.
- **`seasonal_periods=12`** specifies that the seasonal period is 12 months (1 year).
- We're then forecasting future values for the next 12 months using **`result.forecast(forecast_periods)`**.

Running this code will generate a plot showing the original data and the forecasted values for the next 12 months.

## Conclusion

**Project Summary: Exploratory Data Analysis (EDA) of Air Passengers Dataset**

**Dataset Overview:**

- The dataset contains monthly records of international airline passengers from 1949 to 1960.

**Key Findings:**

1. **Trend and Seasonality:**
    - The data exhibits a clear upward trend, indicating an increasing number of airline passengers over the years.
    - Seasonality is evident, with regular spikes in passenger counts around the summer months, indicating a seasonal pattern in travel demand.
2. **Correlation Analysis:**
    - There is a very strong positive correlation (approximately 0.96) between consecutive months. This suggests a high degree of similarity or relationship between the number of passengers in one month and the next.
3. **Forecasting :**
    - A Holt-Winters forecasting model was applied to predict future passenger counts.
    - The model was able to generate forecasted values for the next 12 months, which can provide an estimate of future trends in passenger traffic.

**Conclusion:**

- The analysis of the Air Passengers dataset revealed significant insights into historical trends and patterns in airline passenger traffic.
- The presence of a strong upward trend and seasonality suggests that passenger counts have been consistently increasing over time, with recurring seasonal fluctuations.
- The high correlation between consecutive months further supports the presence of a strong underlying trend.

**Future Considerations:**

- Further analyses or more advanced modeling techniques could be explored to refine the forecasting accuracy.
- Additional data sources or features could be incorporated for a more comprehensive analysis.

**Overall, this project provided valuable insights into the trends and patterns in airline passenger traffic, which can be useful for optimizing resources and making informed business decisions.**

- Entire Code
    
    ```python
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
    ```
    

# Author

[Arjith Praison](https://www.linkedin.com/in/arjith-praison-95b145184/)

University of Siegen
Germany
