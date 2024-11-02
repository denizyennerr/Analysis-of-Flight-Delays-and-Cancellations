# Analysis-of-Flight-Delays-and-Cancellations

## Table Of Contents

- [Project Overview](#project-overview)
- [Project Details](#project-details)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Python Libraries](#python-libraries)
- [Data Cleaning/Preparation](#data-cleaning/preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Analysis](#data-analysis)
- [Code](#code)
- [Recommendations](#recommendations)
- [Limitations](#limitations)
- [References](#references)

## Project Overview
The objective of this project is to analyze a comprehensive dataset containing flight information, including departure and arrival times, delays, distances, weather conditions, and airline details. By exploring this data, we aim to identify significant patterns and correlations that contribute to flight delays. This analysis will help to understand the key factors influencing delays, allowing us to develop predictive models and actionable insights to improve airline punctuality and operational efficiency. Through careful data exploration and modeling, this project seeks to support better decision-making for airlines and enhance the overall passenger experience by reducing delays.

### Project Details
- A date-time index was created for Time Series Analysis and Forecasting, allowing for more effective plotting and time series evaluations. 
- Departure and arrival times were corrected to ensure the accuracy of the data.
- Duplicates were checked to identify any inconsistencies or repeated information within the dataset.
- A logical relationship between weather variables and flight delays was investigated.
- Cleaning and organizing the data from the CSV file.
- Identified which airlines perform the worst in terms of delays.
- Determined which airlines perform the best.
- Analyzed whether flight performance varies by month, whether a certain airline consistently performs poorly, or if performance fluctuates.
- Identified which routes have the highest probability of falling into the level 1 delay category.

### Data Sources
- Airline Flight Delay and Cancellation Data, August 2019 - August 2023 obtained from the US Department of Transportation, Bureau of Transportation Statistics.
- [Download Here](https://www.transtats.bts.gov)


### Tools
- Python

### Python Libraries
- NumPy: A fundamental package for numerical computations in Python.
- Pandas: A library providing high-performance, easy-to-use data structures and data analysis tools for Python.
- Matplotlib: A plotting library for creating static, animated, and interactive visualizations in Python.
- Seaborn: A Python data visualization library based on Matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.

### Data Cleaning/Preparation
In the initial data preparation phase, we performed the following tasks:
1. Creating a date-time variable for time series analysis and the initial data preparation.
2. Correcting departure and arrival times for data preprocessing.
3. Checking for duplicates to ensure data quality.
4. Cleaning the data in the CSV file for accurate analysis.

### Exploratory Data Analysis 

### Data Analysis

- Creating a categorical variable to understand the relationship between each airline and distance.
- Classifying flight distances into three main groups: distances under 500 miles, distances between 500-1000 miles, and distances of 1000 miles and above.
- Identifying the airlines that perform the most flights.
- Comparing the performance of airlines based on delays.
- Analyzing whether airline performance varies by different months of the year and showing the top 5 best and worst airline performances.
- Determining the percentage of flight cancellations and show how it varies by airline.
- Is there a specific time of day and/or time of year when delay durations are higher?
- Identifying the routes with the highest delay durations.

### Code

# Step 1: Import Libraries
We start with importing the necessary libraries.

```Python
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```

# Step 2: Read Dataset
```Python
df = pd.read_csv('/kaggle/input/flights/flights.csv')
df.head()
```
```Python
df_copy= pd.read_csv('/kaggle/input/flights/flights.csv')
df_copy
```
**Dataset Description**

This dataset provides detailed information about flights, including scheduling, delays, airline information, and relevant weather conditions. It spans a range of features that enable an in-depth exploration of factors influencing flight timeliness. Below is a description of the primary features in the dataset:

1. **Date Information**:
   - `year`: Year of the flight.
   - `month`: Month of the flight, allowing for seasonal analysis.
   - `day`: Day of the month the flight was scheduled.

2. **Flight Timing and Delay**:
   - `dep_time`: Actual departure time.
   - `sched_dep_time`: Scheduled departure time.
   - `dep_delay`: Difference between actual and scheduled departure times (in minutes), indicating delays.
   - `arr_time`: Actual arrival time.
   - `sched_arr_time`: Scheduled arrival time.
   - `arr_delay`: Difference between actual and scheduled arrival times (in minutes), indicating arrival delays.

3. **Flight and Airline Information**:
   - `carrier`: The airline carrier code.
   - `flight`: Flight number, identifying each flight uniquely.
   - `tailnum`: Aircraft tail number, unique to each plane.
   - `origin`: Airport code of the departure location.
   - `dest`: Airport code of the destination.
   - `route`: Route of the flight (e.g., SEA-IAH).

4. **Weather Conditions**:
   - `temp`: Temperature (°F) at the departure airport.
   - `dewp`: Dew point (°F), indicating moisture in the air.
   - `humid`: Relative humidity (%) at the departure airport.
   - `wind_dir`: Wind direction in degrees.
   - `wind_speed`: Wind speed (knots) at the departure airport.
   - `wind_gust`: Gust speed (knots) at the departure airport.
   - `precip`: Precipitation amount (inches), indicating rainfall or snowfall.
   - `pressure`: Atmospheric pressure (hPa) at the departure airport.
   - `visib`: Visibility (miles), representing clear or obstructed conditions.
  

# Step 3: Dataset Overview
A comprehensive overview of the dataset:

```
df.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 111006 entries, 0 to 111005
Data columns (total 29 columns):
 #   Column          Non-Null Count   Dtype  
---  ------          --------------   -----  
 0   year            111006 non-null  int64  
 1   month           111006 non-null  int64  
 2   day             111006 non-null  int64  
 3   dep_time        108566 non-null  float64
 4   sched_dep_time  111006 non-null  int64  
 5   dep_delay       108566 non-null  float64
 6   arr_time        108469 non-null  float64
 7   sched_arr_time  111006 non-null  int64  
 8   arr_delay       108332 non-null  float64
 9   carrier         111006 non-null  object 
 10  flight          111006 non-null  int64  
 11  tailnum         110877 non-null  object 
 12  origin          111006 non-null  object 
 13  dest            111006 non-null  object 
 14  air_time        108332 non-null  float64
 15  distance        111006 non-null  int64  
 16  hour            111006 non-null  int64  
 17  minute          111006 non-null  int64  
 18  airline         111006 non-null  object 
 19  route           111006 non-null  object 
 20  temp            111006 non-null  float64
 21  dewp            111006 non-null  float64
 22  humid           111006 non-null  float64
 23  wind_dir        107843 non-null  float64
 24  wind_speed      110727 non-null  float64
 25  wind_gust       110727 non-null  float64
 26  precip          111006 non-null  float64
 27  pressure        111006 non-null  float64
 28  visib           111006 non-null  float64
dtypes: float64(14), int64(9), object(6)
memory usage: 24.6+ MB
```
**Inferences:**

- The dataset comprises **111,006 entries (rows)** and **29 columns**, offering extensive data on various flight characteristics, delays, and weather conditions.
- The columns contain a mix of **data types**:
  - **Integer (`int64`)**: 9 columns, primarily representing time components, distance, and identifiers.
  - **Float (`float64`)**: 14 columns, which include precise timing data, delay metrics, and weather-related variables.
  - **Object**: 6 columns, usually representing categorical data such as airline carriers, route information, and airport codes.
- The dataset has some **missing values** in specific columns:
  - **Timing and Delay Information**: Columns such as `dep_time`, `dep_delay`, `arr_time`, `arr_delay`, and `air_time` have missing values, suggesting that not all flights have complete timing and delay data.
  - **Aircraft Identifier**: The `tailnum` column also has some missing entries, indicating gaps in identifying specific aircraft for certain flights.
  - **Weather Variables**: Columns such as `wind_dir`, `wind_speed`, and `wind_gust` contain missing values, likely due to unavailable weather data at certain times.


## Step 3.2 Summary Statistics for Numerical Variables

| Variable      | Count     | Mean       | Std       | Min   | 25%       | 50%       | 75%       | Max       |
|---------------|-----------|------------|-----------|-------|-----------|-----------|-----------|-----------|
| year          | 111006.0  | 2022.000000 | 0.000000  | 2022.00 | 2022.000000 | 2022.000000 | 2022.000000 | 2022.000000 |
| month         | 111006.0  | 3.623264   | 1.713287  | 1.00  | 2.000000  | 4.000000  | 5.000000  | 6.000000  |
| day           | 111006.0  | 15.736654  | 8.723487  | 1.00  | 8.000000  | 16.000000 | 23.000000 | 31.000000 |
| dep_time      | 108566.0  | 1336.449487| 540.664793| 1.00  | 912.000000| 1309.000000 | 1800.000000 | 2400.000000 |
| sched_dep_time| 111006.0  | 1342.068158| 530.610560| 2.00  | 910.000000| 1305.000000 | 1800.000000 | 2359.000000 |
| dep_delay     | 108566.0  | 8.038456   | 41.687738 | -36.00 | -5.000000 | -2.000000 | 6.000000  | 2120.000000 |
| arr_time      | 108469.0  | 1467.243636| 570.938019| 1.00  | 1055.000000 | 1520.000000 | 1917.000000 | 2400.000000 |
| sched_arr_time| 111006.0  | 1506.330090| 546.796862| 3.00  | 1115.000000 | 1540.000000 | 1937.000000 | 2359.000000 |
| arr_delay     | 108332.0  | 2.359524   | 43.088734 | -65.00 | -14.000000 | -5.000000 | 6.000000  | 2098.000000 |
| flight        | 111006.0  | 433.663856 | 267.779635| 1.00  | 210.000000 | 416.000000 | 644.000000 | 1100.000000 |
| air_time      | 108332.0  | 136.527628 | 82.487434 | 17.00 | 78.000000  | 120.000000 | 193.000000 | 397.000000 |
| distance      | 111006.0  | 1068.619183| 746.859903| 93.00 | 543.000000 | 909.000000 | 1542.000000 | 2724.000000 |
| hour          | 111006.0  | 13.135614  | 5.269788  | 0.00  | 9.000000   | 13.000000  | 18.000000  | 23.000000 |
| minute        | 111006.0  | 28.506729  | 18.354465 | 0.00  | 15.000000  | 30.000000  | 45.000000  | 59.000000 |
| temp          | 111006.0  | 48.155538  | 9.363201  | 21.90 | 42.000000  | 47.000000  | 54.000000  | 99.000000 |
| dewp          | 111006.0  | 40.454927  | 7.968649  | 1.00  | 36.000000  | 40.000000  | 46.000000  | 61.000000 |
| humid         | 111006.0  | 76.506300  | 15.192175 | 15.79 | 68.460000  | 79.400000  | 88.670000  | 100.000000 |
| wind_dir      | 107843.0  | 164.500524 | 101.084372| 0.00  | 100.000000 | 180.000000 | 220.000000 | 360.000000 |
| wind_speed    | 110727.0  | 6.995307   | 4.507431  | 0.00  | 4.603120   | 6.904680   | 9.206240   | 27.618720 |
| wind_gust     | 110727.0  | 8.050059   | 5.187061  | 0.00  | 5.297178   | 7.945768   | 10.594357  | 31.783071 |
| precip        | 111006.0  | 0.005774   | 0.022099  | 0.00  | 0.000000   | 0.000000   | 0.000000   | 0.320000  |
| pressure      | 111006.0  | 1020.187948| 7.710084  | 991.00 | 1015.200000 | 1020.700000 | 1025.500000 | 1039.200000 |
| visib         | 111006.0  | 8.922346   | 2.537055  | 0.00  | 10.000000  | 10.000000  | 10.000000  | 10.000000 |


**Inferences:**

- **year**: All records are from the year 2022, indicating no variation in the year feature.
- **month, day**: These columns represent the date of the scheduled flight. `month` varies from January to June, while `day` spans from 1 to 31, suggesting a good distribution across months and days.
- **hour, minute**: These features represent the scheduled departure hour and minute. They cover the full range of hours (0–23) and minutes (0–59), reflecting departures at various times throughout the day.
- **dep_time, sched_dep_time, arr_time, sched_arr_time**: These columns capture actual and scheduled departure and arrival times in 24-hour format. They cover a broad range, indicating flights scheduled and departing throughout the day and night.
- **dep_delay, arr_delay**: Key target variables showing departure and arrival delays in minutes. Values range from negative (early departure/arrival) to positive (late departure/arrival), with `dep_delay` reaching a maximum of 2120 minutes and `arr_delay` up to 2098 minutes.
- **flight**: This is the flight number, which spans from 1 to 1100, indicating a wide range of flight identifiers.
- **air_time**: Flight duration, with values ranging from 17 to 397 minutes, showing the variety in flight lengths within the dataset.
- **distance**: Total distance between origin and destination airports, ranging from 93 to 2724 miles, capturing flights of different lengths.
- **temp, dewp**: Temperature (`temp`) varies from 21.9 to 99°F, and dew point (`dewp`) from 1 to 61°F, reflecting seasonal weather conditions at departure.
- **humid**: Relative humidity ranges from 15.79% to 100%, indicating varied atmospheric moisture conditions.
- **wind_dir, wind_speed, wind_gust**: Wind direction ranges from 0 to 360 degrees, while wind speed and gust vary up to 27.6 and 31.8 knots, respectively, reflecting a range of wind conditions at departure.
- **precip**: Precipitation varies from 0 to 0.32 inches, with many zero entries, suggesting limited rainfall data.
- **pressure**: Atmospheric pressure varies between 991 and 1039.2 hPa, with a median of around 1020.7 hPa.
- **visib**: Visibility ranges from 0 to 10 miles, with the majority at 10 miles, indicating mostly clear conditions.

## Step 3.3 | Summary Statistics for Categorical Variables

```
df.describe(include='object')
```

| Metric   | carrier | tailnum | origin | dest | airline             | route   |
|----------|---------|---------|--------|------|----------------------|---------|
| count    | 111006  | 110877  | 111006 | 111006 | 111006             | 111006  |
| unique   | 12      | 3520    | 2      | 97   | 12                 | 149     |
| top      | AS      | N447QX  | SEA    | LAX  | Alaska Airlines Inc.| SEA-PDX |
| freq     | 41697   | 524     | 82559  | 5450 | 41697              | 3867    |

**Inferences for Categorical Variables:**

- **carrier, airline**: These columns represent the airline carrier code (`carrier`) and the full airline name (`airline`). There are 12 unique airlines in the dataset. The most frequent carrier is Alaska Airlines Inc. (`AS`), accounting for 41,697 flights.
- **tailnum**: This column is a unique identifier for each aircraft, with a total of 3,520 unique aircraft in the dataset. The most frequent aircraft identifier is `N447QX`, appearing in 524 flights.
- **origin, dest**: These columns represent the airport codes for the departure (`origin`) and arrival (`dest`) airports. There are 2 unique origin airports, with Seattle (SEA) being the most frequent departure airport (82,559 flights). For destination airports, there are 97 unique values, with Los Angeles (LAX) being the most common destination, totaling 5,450 flights.
- **route**: This column represents the route of the flight, combining origin and destination codes (e.g., `SEA-PDX`). There are 149 unique routes, with the Seattle to Portland (SEA-PDX) route being the most frequent, appearing in 3,867 flights.



# Exploratory Data Analysis

## Step 4.1 Univariate Analysis
We can perform univariate analysis on these columns based on their datatype:
For numerical data, we can use a histogram to visualize the data distribution. The number of bins should be chosen appropriately to represent the data well.
For categorical data, we can use a bar plot to visualize the frequency of each category.

```Python
import pandas as pd
import matplotlib.pyplot as plt

# List of columns based on data types
numerical_columns = ['year', 'month', 'day', 'sched_dep_time', 'sched_arr_time', 'flight', 'distance', 'hour', 'minute',
                     'dep_time', 'dep_delay', 'arr_time', 'arr_delay', 'air_time', 'temp', 'dewp', 'humid', 
                     'wind_dir', 'wind_speed', 'wind_gust', 'precip', 'pressure', 'visib']
categorical_columns = ['carrier', 'tailnum', 'origin', 'dest', 'airline', 'route']

# Univariate analysis for numerical columns
for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    plt.hist(df[col].dropna(), bins=30, edgecolor='k', alpha=0.7)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Univariate analysis for categorical columns
for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    df[col].value_counts().plot(kind='bar', color='darkblue', edgecolor='k')
    plt.title(f'Bar Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

```
## Step 4.2 Bivariate Analysis
For our bivariate analysis, we'll consider the dep_delay column as the target. 
We can analyze the relationship between dep_delay and other columns. 
To do this, we can use scatter plots for numerical columns and violin plots for categorical columns. 
We skip id, flight, tailnum, time_hour as they are identifiers or contain redundant information.

```
# Set the target column
target_column = 'dep_delay'  

# Numerical columns (excluding identifiers and redundant info)
numerical_columns = ['year', 'month', 'day', 'sched_dep_time', 'sched_arr_time', 'distance', 'hour', 
                     'minute', 'air_time', 'temp', 'dewp', 'humid', 'wind_dir', 'wind_speed', 
                     'wind_gust', 'precip', 'pressure', 'visib']

# Scatter plots for numerical columns vs target
for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=df, x=col, y=target_column, alpha=0.5)
    plt.title(f'Scatter Plot of {col} vs {target_column}')
    plt.xlabel(col)
    plt.ylabel(target_column)
    plt.show()

# Categorical columns (excluding identifiers and redundant info)
categorical_columns = ['carrier', 'origin', 'dest', 'airline', 'route']

# Violin plots for categorical columns vs target
for col in categorical_columns:
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df, x=col, y=target_column, inner="quartile", color="brown")
    plt.title(f'Violin Plot of {target_column} by {col}')
    plt.xlabel(col)
    plt.ylabel(target_column)
    plt.xticks(rotation=45)
    plt.show()

```

### Project Details
- A date-time index was created for Time Series Analysis and Forecasting, allowing for more effective plotting and time series evaluations. 
- Departure and arrival times were corrected to ensure the accuracy of the data.
- Duplicates were checked to identify any inconsistencies or repeated information within the dataset.
- A logical relationship between weather variables and flight delays was investigated.
- Cleaning and organizing the data from the CSV file.
- Identified which airlines perform the worst in terms of delays.
- Determined which airlines perform the best.
- Analyzed whether flight performance varies by month, whether a certain airline consistently performs poorly, or if performance fluctuates.
- Identified which routes have the highest probability of falling into the level 1 delay category.

  
A date-time index was generated for Time Series Analysis and Forecasting, enabling more effective visualization and time series analysis.
```Python
df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
df.insert(0, 'date', df.pop('date'))
df.drop(columns=['year', 'month', 'day','hour', 'minute'], inplace=True)
df['sched_dep_time'] = df['sched_dep_time'].astype(str).str.zfill(4)
df['sched_dep_time']= pd.to_datetime(df['sched_dep_time'], format="%H%M").dt.time.astype(str).str[:5]
df.head(30)
```
Interference:

1. **id**: The `id` feature is a unique identifier for each flight record. It has no informational value for predicting delays, as it merely distinguishes each record without providing insight into any flight characteristics. Including `id` could add noise and increase computational complexity, so it will be excluded.
2. **tailnum**: Each aircraft has a unique `tailnum`, resulting in a high number of unique values (4043) across the dataset. Although specific aircraft might be prone to delays (e.g., older planes requiring frequent maintenance), the high dimensionality of `tailnum` could lead to overfitting. The potential predictive gain from including `tailnum` is outweighed by the risk of overfitting, so it will be removed.
3. **time_hour**: This feature represents the scheduled departure time in a "yyyy-mm-dd hh:mm:ss" format. Since we have separate features for `year`, `month`, `day`, and `sched_dep_time`, the `time_hour` feature is redundant and will be removed to reduce data redundancy.
4. **minute**: The `minute` component of the scheduled departure time is already captured within `sched_dep_time`. Including both `minute` and `sched_dep_time` would introduce redundancy without adding unique information, so `minute` will be excluded.
5. **hour**: Similar to `minute`, `hour` is also part of the `sched_dep_time` feature. As it does not contribute additional information beyond what `sched_dep_time` provides, it will be removed to avoid redundancy.
6. **carrier**: The `carrier` feature represents a two-letter airline code, while the `name` feature contains the full name of the airline. Since these features provide overlapping information, we will retain `name` (as it is more descriptive) and remove `carrier` to avoid redundancy.
7. **year**: Since all flights in this dataset took place in 2013, `year` is a constant feature and does not contribute to the model’s predictive power. A constant feature cannot help the model distinguish between records, so it will be removed.
8. **flight**: The `flight` feature, representing designated flight numbers, has many unique values (3844). Although specific flights might have delay patterns, the high dimensionality of `flight` could lead to overfitting. Consequently, `flight` will be excluded from the dataset to maintain model generalizability.

# Step 5: Data Preprocessing
# Step 5.1: Missing Value Treatment

```
df['sched_dep_time'] = df['sched_dep_time'].astype(str).str.zfill(4)
df['sched_dep_time']= pd.to_datetime(df['sched_dep_time'], format="%H%M").dt.time.astype(str).str[:5]
df.head(7)
```
| date                | dep_time | sched_dep_time | dep_delay | arr_time | sched_arr_time | arr_delay | carrier | flight | tailnum | origin | dest | air_time | distance | airline               | route   | temp | dewp | humid | wind_dir | wind_speed | wind_gust | precip | pressure | visib |
|---------------------|----------|----------------|-----------|----------|----------------|-----------|---------|--------|---------|--------|------|----------|----------|-----------------------|---------|------|------|-------|----------|------------|-----------|--------|----------|-------|
| 2022-01-01 23:59:00 | 1.0      | 23:59         | 2.0       | 604.0    | 618            | -14.0     | UA      | 555    | N405UA  | SEA    | IAH  | 221.0    | 1874     | United Air Lines Inc. | SEA-IAH | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 22:50:00 | 1.0      | 22:50         | 71.0      | 242.0    | 142            | 60.0      | AS      | 72     | N265AK  | SEA    | FAI  | 193.0    | 1533     | Alaska Airlines Inc.  | SEA-FAI | 32.0 | 23.0 | 69.04 | 170.0    | 9.20624    | 10.594357 | 0.0    | 1023.4   | 10.0  |
| 2022-01-01 23:55:00 | 10.0     | 23:55         | 15.0      | 759.0    | 730            | 29.0      | AS      | 270    | N274AK  | SEA    | ATL  | 261.0    | 2182     | Alaska Airlines Inc.  | SEA-ATL | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 23:50:00 | 25.0     | 23:50         | 35.0      | 606.0    | 550            | 16.0      | AS      | 7      | N281AK  | SEA    | ORD  | 193.0    | 1721     | Alaska Airlines Inc.  | SEA-ORD | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 23:49:00 | 35.0     | 23:49         | 46.0      | 616.0    | 545            | 31.0      | UA      | 507    | N426UA  | PDX    | ORD  | 196.0    | 1739     | United Air Lines Inc. | PDX-ORD | 33.0 | 19.0 | 55.75 | 120.0    | 6.90468    | 7.945768  | 0.0    | 1025.1   | 10.0  |
| 2022-01-01 23:52:00 | 51.0     | 23:52         | 59.0      | 840.0    | 758            | 42.0      | B6      | 366    | N625JB  | PDX    | JFK  | 269.0    | 2454     | JetBlue Airways       | PDX-JFK | 33.0 | 19.0 | 55.75 | 120.0    | 6.90468    | 7.945768  | 0.0    | 1025.1   | 10.0  |
| 2022-01-01 00:43:00 | 104.0    | 00:43         | 21.0      | 936.0    | 930            | 6.0       | AA      | 501    | N413AN  | SEA    | MIA  | 312.0    | 2724     | American Airlines Inc.| SEA-MIA | 25.0 | 14.0 | 62.50 | 350.0    | 8.05546    | 9.270062  | 0.0    | 1020.7   | 10.0  |


```
df['sched_arr_time'] = df['sched_arr_time'].astype(str).str.zfill(4)
df['sched_arr_time']= pd.to_datetime(df['sched_arr_time'], format="%H%M").dt.time.astype(str).str[:5]
df.head()
```

| date                | dep_time | sched_dep_time | dep_delay | arr_time | sched_arr_time | arr_delay | carrier | flight | tailnum | origin | dest | air_time | distance | airline               | route   | temp | dewp | humid | wind_dir | wind_speed | wind_gust | precip | pressure | visib |
|---------------------|----------|----------------|-----------|----------|----------------|-----------|---------|--------|---------|--------|------|----------|----------|-----------------------|---------|------|------|-------|----------|------------|-----------|--------|----------|-------|
| 2022-01-01 23:59:00 | 1.0      | 23:59         | 2.0       | 604.0    | 06:18          | -14.0     | UA      | 555    | N405UA  | SEA    | IAH  | 221.0    | 1874     | United Air Lines Inc. | SEA-IAH | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 22:50:00 | 1.0      | 22:50         | 71.0      | 242.0    | 01:42          | 60.0      | AS      | 72     | N265AK  | SEA    | FAI  | 193.0    | 1533     | Alaska Airlines Inc.  | SEA-FAI | 32.0 | 23.0 | 69.04 | 170.0    | 9.20624    | 10.594357 | 0.0    | 1023.4   | 10.0  |
| 2022-01-01 23:55:00 | 10.0     | 23:55         | 15.0      | 759.0    | 07:30          | 29.0      | AS      | 270    | N274AK  | SEA    | ATL  | 261.0    | 2182     | Alaska Airlines Inc.  | SEA-ATL | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 23:50:00 | 25.0     | 23:50         | 35.0      | 606.0    | 05:50          | 16.0      | AS      | 7      | N281AK  | SEA    | ORD  | 193.0    | 1721     | Alaska Airlines Inc.  | SEA-ORD | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 23:49:00 | 35.0     | 23:49         | 46.0      | 616.0    | 05:45          | 31.0      | UA      | 507    | N426UA  | PDX    | ORD  | 196.0    | 1739     | United Air Lines Inc. | PDX-ORD | 33.0 | 19.0 | 55.75 | 120.0    | 6.90468    | 7.945768  | 0.0    | 1025.1   | 10.0  |

```
df['arr_time'] = pd.to_numeric(df['arr_time'], errors='coerce')
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['arr_time'])
df['arr_time'] = df['arr_time'].astype(int)
df.head(6)
```
| date                | dep_time | sched_dep_time | dep_delay | arr_time | sched_arr_time | arr_delay | carrier | flight | tailnum | origin | dest | air_time | distance | airline               | route   | temp | dewp | humid | wind_dir | wind_speed | wind_gust | precip | pressure | visib |
|---------------------|----------|----------------|-----------|----------|----------------|-----------|---------|--------|---------|--------|------|----------|----------|-----------------------|---------|------|------|-------|----------|------------|-----------|--------|----------|-------|
| 2022-01-01 23:59:00 | 1.0      | 23:59         | 2.0       | 604      | 06:18          | -14.0     | UA      | 555    | N405UA  | SEA    | IAH  | 221.0    | 1874     | United Air Lines Inc. | SEA-IAH | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 22:50:00 | 1.0      | 22:50         | 71.0      | 242      | 01:42          | 60.0      | AS      | 72     | N265AK  | SEA    | FAI  | 193.0    | 1533     | Alaska Airlines Inc.  | SEA-FAI | 32.0 | 23.0 | 69.04 | 170.0    | 9.20624    | 10.594357 | 0.0    | 1023.4   | 10.0  |
| 2022-01-01 23:55:00 | 10.0     | 23:55         | 15.0      | 759      | 07:30          | 29.0      | AS      | 270    | N274AK  | SEA    | ATL  | 261.0    | 2182     | Alaska Airlines Inc.  | SEA-ATL | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 23:50:00 | 25.0     | 23:50         | 35.0      | 606      | 05:50          | 16.0      | AS      | 7      | N281AK  | SEA    | ORD  | 193.0    | 1721     | Alaska Airlines Inc.  | SEA-ORD | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 23:49:00 | 35.0     | 23:49         | 46.0      | 616      | 05:45          | 31.0      | UA      | 507    | N426UA  | PDX    | ORD  | 196.0    | 1739     | United Air Lines Inc. | PDX-ORD | 33.0 | 19.0 | 55.75 | 120.0    | 6.90468    | 7.945768  | 0.0    | 1025.1   | 10.0  |
| 2022-01-01 23:52:00 | 51.0     | 23:52         | 59.0      | 840      | 07:58          | 42.0      | B6      | 366    | N625JB  | PDX    | JFK  | 269.0    | 2454     | JetBlue Airways       | PDX-JFK | 33.0 | 19.0 | 55.75 | 120.0    | 6.90468    | 7.945768  | 0.0    | 1025.1   | 10.0  |




















### Results/Findings 
The analysis results are summarised as follows:
1. 
2. 
3. 


### Recommendations
Based on the analysis, we recommend the following actions:


### Limitations


### References
