# Analysis-of-Flight-Delays-and-Cancellations

## Table Of Contents

- [Project Overview](#project-overview)
- [Project Details](#project-details)
- [Project Questions](#project-questions)
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

### Project Questions:
1. Create a categorical variable to understand the relationship between each airline and flight distance.
2. Classify flight distances into three main groups: distances under 500 miles, distances between 500-1000 miles, and distances of 1000 miles and above.
3. Identify the airlines with the highest number of flights conducted.
4. Compare the performance of airlines based on delays.
5. Examine whether airline performance varies across different months of the year, and show the top 5 airlines with the best performance and the bottom 5 with the worst performance.
6. Determine the percentage of flight cancellations and display how this varies by airline.
7. Is there a specific time of day and/or time of year with higher delay durations?
8. Which routes have the highest delay durations?

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

## Exploratory Data Analysis 

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



# Step 4: Exploratory Data Analysis

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

```Python
import pandas as pd
import matplotlib.pyplot as plt

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

# Step 5: Data Preprocessing
## Step 5.1: Missing Value Treatment

```Python
df.columns
```
# Flight Dataset Columns

The dataset contains information on flights, including schedule, delay, and weather data. Below is a list of columns included:

| Column            | Description                                              |
|-------------------|----------------------------------------------------------|
| **year**          | Year of the flight (e.g., 2022)                          |
| **month**         | Month of the flight (1-12)                               |
| **day**           | Day of the month (1-31)                                  |
| **dep_time**      | Actual departure time (in HHMM format)                   |
| **sched_dep_time**| Scheduled departure time (in HHMM format)                |
| **dep_delay**     | Departure delay in minutes                               |
| **arr_time**      | Actual arrival time (in HHMM format)                     |
| **sched_arr_time**| Scheduled arrival time (in HHMM format)                  |
| **arr_delay**     | Arrival delay in minutes                                 |
| **carrier**       | Airline carrier code (e.g., AA for American Airlines)    |
| **flight**        | Flight number                                            |
| **tailnum**       | Aircraft tail number                                     |
| **origin**        | Origin airport code                                      |
| **dest**          | Destination airport code                                 |
| **air_time**      | Time spent in the air (in minutes)                       |
| **distance**      | Distance between origin and destination (in miles)       |
| **hour**          | Hour of scheduled departure (24-hour format)             |
| **minute**        | Minute of scheduled departure                            |
| **airline**       | Full airline name                                        |
| **route**         | Route from origin to destination (origin-destination)    |
| **temp**          | Temperature at departure airport (in °F)                 |
| **dewp**          | Dew point temperature (in °F)                            |
| **humid**         | Humidity percentage                                      |
| **wind_dir**      | Wind direction (in degrees)                              |
| **wind_speed**    | Wind speed (in mph)                                      |
| **wind_gust**     | Wind gust speed (in mph)                                 |
| **precip**        | Precipitation amount (in inches)                         |
| **pressure**      | Atmospheric pressure (in millibars)                      |
| **visib**         | Visibility (in miles)                                    |
```Python
df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])

df.insert(0, 'date', df.pop('date'))

df.drop(columns=['year', 'month', 'day','hour', 'minute'], inplace=True)
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


The following code helps us to perform data preprocessing to clean and standardize the time-related columns in the dataset. Here's a breakdown of each step:

1. **Scheduled Departure Time (`sched_dep_time`)**: 
   - Converts `sched_dep_time` to a 4-digit string format (`HHMM`) using `str.zfill(4)`, ensuring all times have leading zeros as necessary.
   - Parses the formatted string into a time format (`HH:MM`) for easier interpretation and analysis.

```Python
df['sched_dep_time'] =df['sched_dep_time'].astype(str).str.zfill(4)
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


2. **Scheduled Arrival Time (`sched_arr_time`)**:
   - Similar to `sched_dep_time`, this converts `sched_arr_time` to a standardized 4-digit string and then to a readable time format (`HH:MM`).
  
```Python
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


3. **Arrival Time (`arr_time`)**:
   - Converts `arr_time` to a numeric type, coercing any non-numeric values to `NaN`.
   - Replaces any infinity values with `NaN` and drops rows where `arr_time` is `NaN`, ensuring only valid numeric times remain.
   - Converts `arr_time` to an integer for consistency in the data format.

This preprocessing improves data quality, making the `sched_dep_time`, `sched_arr_time`, and `arr_time` columns uniformly formatted and ready for analysis.    

```Python
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

The following code refines the `arr_time` column by ensuring that all values are in a standardized time format (`HH:MM`), while handling potential formatting inconsistencies. Here’s a summary of each step:

1. **Standardize Format**:
   - Converts `arr_time` to a 4-digit string format using `str.zfill(4)`, ensuring times are uniformly formatted (e.g., `0300` for 3:00 AM).
2. **Remove Extra Whitespace**:
   - Strips any leading or trailing whitespace from the `arr_time` values.
3. **Filter Numeric Values**:
   - Filters the `arr_time` column to keep only numeric entries, removing any rows with non-numeric values that could cause parsing errors.
4. **Convert to Time Format**:
   - Attempts to parse `arr_time` into a time format (`HH:MM`) using `pd.to_datetime`, setting invalid formats to `NaT` (`errors='coerce'`).
   - Extracts and formats the time as a string in `HH:MM` format, making it ready for consistent time-based analysis.

The code improves data integrity by ensuring `arr_time` is uniformly formatted as `HH:MM` and excludes any non-time values, making the dataset cleaner and more reliable.

```Python
df['arr_time']=df['arr_time'].astype(str).str.zfill(4)
df['arr_time']

df['arr_time']=df['arr_time'].astype(str).str.zfill(4)
df['arr_time']

df['arr_time'] = df['arr_time'].str.strip()
df_filtered = df[df['arr_time'].str.isnumeric()]

try:
  df['arr_time'] = pd.to_datetime(df_filtered['arr_time'], format="%H%M", errors='coerce')
except ValueError:
  pass

df['arr_time'] = df['arr_time'].dt.time.astype(str).str[:5]

df.head()
```

| date                | dep_time | sched_dep_time | dep_delay | arr_time | sched_arr_time | arr_delay | carrier | flight | tailnum | route   | temp | dewp | humid | wind_dir | wind_speed | wind_gust | precip | pressure | visib |
|---------------------|----------|----------------|-----------|----------|----------------|-----------|---------|--------|---------|---------|------|------|-------|----------|------------|-----------|--------|----------|-------|
| 2022-01-01 23:59:00 | 00:01    | 23:59         | 2.0       | NaT      | 06:18          | -14.0     | UA      | 555    | N405UA  | SEA-IAH | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 22:50:00 | 00:01    | 22:50         | 71.0      | NaT      | 01:42          | 60.0      | AS      | 72     | N265AK  | SEA-FAI | 32.0 | 23.0 | 69.04 | 170.0    | 9.20624    | 10.594357 | 0.0    | 1023.4   | 10.0  |
| 2022-01-01 23:55:00 | 00:10    | 23:55         | 15.0      | NaT      | 07:30          | 29.0      | AS      | 270    | N274AK  | SEA-ATL | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 23:50:00 | 00:25    | 23:50         | 35.0      | NaT      | 05:50          | 16.0      | AS      | 7      | N281AK  | SEA-ORD | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 23:49:00 | 00:35    | 23:49         | 46.0      | NaT      | 05:45          | 31.0      | UA      | 507    | N426UA  | PDX-ORD | 33.0 | 19.0 | 55.75 | 120.0    | 6.90468    | 7.945768  | 0.0    | 1025.1   | 10.0  |

The following code standardizes the `dep_time` (departure time) column by ensuring all values are formatted as valid times in the `HH:MM` format. Here’s a breakdown of the steps:

1. **Convert and Pad Values**:
   - Converts `dep_time` to an integer, then back to a string, using `str.zfill(4)` to ensure all values have four digits (e.g., `0800` for 8:00 AM).
2. **Remove Extra Whitespace**:
   - Strips any leading or trailing whitespace from the `dep_time` values for consistency.
3. **Filter Numeric Values**:
   - Filters the `dep_time` column to retain only numeric entries, which helps prevent parsing errors by excluding rows with non-numeric values.
4. **Convert to Time Format**:
   - Attempts to parse `dep_time` into a time format (`HH:MM`) using `pd.to_datetime`, setting invalid formats to `NaT` (`errors='coerce'`).
   - Extracts and formats the time as a string (`HH:MM`), finalizing the column for accurate and uniform time-based analysis.

This process improves data consistency and reliability by ensuring `dep_time` is correctly formatted as `HH:MM`, making it suitable for further analysis or modeling.

```Python
df['dep_time']=df['dep_time'].astype(int).astype(str).str.zfill(4)
df['dep_time']

df['dep_time']=df['dep_time'].astype(int).astype(str).str.zfill(4)
df['dep_time']

df['dep_time'] = df['dep_time'].str.strip()

df_filtered = df[df['dep_time'].str.isnumeric()]

try:
  df['dep_time'] = pd.to_datetime(df_filtered['dep_time'], format="%H%M", errors='coerce')
except ValueError:
  pass

df['dep_time'] = df['dep_time'].dt.time.astype(str).str[:5]
df.head()

```

| date                | dep_time | sched_dep_time | dep_delay | arr_time | sched_arr_time | arr_delay | carrier | flight | tailnum | route   | temp | dewp | humid | wind_dir | wind_speed | wind_gust | precip | pressure | visib |
|---------------------|----------|----------------|-----------|----------|----------------|-----------|---------|--------|---------|---------|------|------|-------|----------|------------|-----------|--------|----------|-------|
| 2022-01-01 23:59:00 | 00:01    | 23:59          | 2.0       | 06:04    | 06:18          | -14.0     | UA      | 555    | N405UA  | SEA-IAH | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 22:50:00 | 00:01    | 22:50          | 71.0      | 02:42    | 01:42          | 60.0      | AS      | 72     | N265AK  | SEA-FAI | 32.0 | 23.0 | 69.04 | 170.0    | 9.20624    | 10.594357 | 0.0    | 1023.4   | 10.0  |
| 2022-01-01 23:55:00 | 00:10    | 23:55          | 15.0      | 07:59    | 07:30          | 29.0      | AS      | 270    | N274AK  | SEA-ATL | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 23:50:00 | 00:25    | 23:50          | 35.0      | 06:06    | 05:50          | 16.0      | AS      | 7      | N281AK  | SEA-ORD | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 23:49:00 | 00:35    | 23:49          | 46.0      | 06:16    | 05:45          | 31.0      | UA      | 507    | N426UA  | PDX-ORD | 33.0 | 19.0 | 55.75 | 120.0    | 6.90468    | 7.945768  | 0.0    | 1025.1   | 10.0  |

The following code helps us to performs data cleaning for the `arr_delay` (arrival delay) and `dep_delay` (departure delay) columns, ensuring that any missing values are handled and the data types are standardized for analysis. Here’s an explanation:

1. **Handle Missing Values**:
   - For both `arr_delay` and `dep_delay`, any `NaN` values are replaced with `0` using `.fillna(0)`. This approach assumes that missing delay values represent no delay.
2. **Convert to Integer**:
   - Converts both `arr_delay` and `dep_delay` columns to integer type, which standardizes the data type and ensures compatibility with calculations or visualizations requiring integer values.
3. **Display Dataframe Settings**:
   - `pd.set_option('display.max_columns', None)` ensures that all columns of the dataframe are displayed when calling `.head()`, providing a full view of the dataset’s structure and confirming the transformations.

This code improves the dataset's reliability by treating missing delays as zero and enforcing integer formatting, making it more consistent and suitable for further analysis.

```Python
df['arr_delay'] = df['arr_delay'].fillna(0)

df['arr_delay'] = df['arr_delay'].astype(int)
df['arr_delay']

df['dep_delay'] = df['dep_delay'].fillna(0)

df['dep_delay'] = df['dep_delay'].astype(int)
df['dep_delay']
```
```
pd.set_option('display.max_columns', None)
df.head()
```
| date                | dep_time | sched_dep_time | dep_delay | arr_time | sched_arr_time | arr_delay | carrier | flight | tailnum | origin | dest | air_time | distance | airline               | route   | temp | dewp | humid | wind_dir | wind_speed | wind_gust | precip | pressure | visib |
|---------------------|----------|----------------|-----------|----------|----------------|-----------|---------|--------|---------|--------|------|----------|----------|-----------------------|---------|------|------|-------|----------|------------|-----------|--------|----------|-------|
| 2022-01-01 23:59:00 | 00:01    | 23:59         | 2         | NaT      | 06:18          | -14       | UA      | 555    | N405UA  | SEA    | IAH  | 221.0    | 1874     | United Air Lines Inc. | SEA-IAH | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 22:50:00 | 00:01    | 22:50         | 71        | NaT      | 01:42          | 60        | AS      | 72     | N265AK  | SEA    | FAI  | 193.0    | 1533     | Alaska Airlines Inc.  | SEA-FAI | 32.0 | 23.0 | 69.04 | 170.0    | 9.20624    | 10.594357 | 0.0    | 1023.4   | 10.0  |
| 2022-01-01 23:55:00 | 00:10    | 23:55         | 15        | NaT      | 07:30          | 29        | AS      | 270    | N274AK  | SEA    | ATL  | 261.0    | 2182     | Alaska Airlines Inc.  | SEA-ATL | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 23:50:00 | 00:25    | 23:50         | 35        | NaT      | 05:50          | 16        | AS      | 7      | N281AK  | SEA    | ORD  | 193.0    | 1721     | Alaska Airlines Inc.  | SEA-ORD | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 2022-01-01 23:49:00 | 00:35    | 23:49         | 46        | NaT      | 05:45          | 31        | UA      | 507    | N426UA  | PDX    | ORD  | 196.0    | 1739     | United Air Lines Inc. | PDX-ORD | 33.0 | 19.0 | 55.75 | 120.0    | 6.90468    | 7.945768  | 0.0    | 1025.1   | 10.0  |


## Step 5.2: Irrelevant Feature Removal

```Python
duplicates = df.duplicated()
duplicates
```
```Python
duplicate_rows = df[duplicates]
duplicate_rows
```

```Python
df_cleaned = df.drop_duplicates()
df_cleaned
```

| Index | date                | dep_time | sched_dep_time | dep_delay | arr_time | sched_arr_time | arr_delay | carrier | flight | tailnum | origin | dest | air_time | distance | airline               | route   | temp | dewp | humid | wind_dir | wind_speed | wind_gust | precip | pressure | visib |
|-------|----------------------|----------|----------------|-----------|----------|----------------|-----------|---------|--------|---------|--------|------|----------|----------|-----------------------|---------|------|------|-------|----------|------------|-----------|--------|----------|-------|
| 0     | 2022-01-01 23:59:00 | 00:01    | 23:59         | 2         | 06:04    | 06:18          | -14       | UA      | 555    | N405UA  | SEA    | IAH  | 221.0    | 1874     | United Air Lines Inc. | SEA-IAH | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 1     | 2022-01-01 22:50:00 | 00:01    | 22:50         | 71        | 02:42    | 01:42          | 60        | AS      | 72     | N265AK  | SEA    | FAI  | 193.0    | 1533     | Alaska Airlines Inc.  | SEA-FAI | 32.0 | 23.0 | 69.04 | 170.0    | 9.20624    | 10.594357 | 0.0    | 1023.4   | 10.0  |
| 2     | 2022-01-01 23:55:00 | 00:10    | 23:55         | 15        | 07:59    | 07:30          | 29        | AS      | 270    | N274AK  | SEA    | ATL  | 261.0    | 2182     | Alaska Airlines Inc.  | SEA-ATL | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 3     | 2022-01-01 23:50:00 | 00:25    | 23:50         | 35        | 06:06    | 05:50          | 16        | AS      | 7      | N281AK  | SEA    | ORD  | 193.0    | 1721     | Alaska Airlines Inc.  | SEA-ORD | 33.0 | 23.0 | 66.06 | 160.0    | 8.05546    | 9.270062  | 0.0    | 1022.9   | 10.0  |
| 4     | 2022-01-01 23:49:00 | 00:35    | 23:49         | 46        | 06:16    | 05:45          | 31        | UA      | 507    | N426UA  | PDX    | ORD  | 196.0    | 1739     | United Air Lines Inc. | PDX-ORD | 33.0 | 19.0 | 55.75 | 120.0    | 6.90468    | 7.945768  | 0.0    | 1025.1   | 10.0  |


# Step 6: Project Details

- A date-time index was created for Time Series Analysis and Forecasting, allowing for more effective plotting and time series evaluations. 
- Departure and arrival times were corrected to ensure the accuracy of the data.
- Duplicates were checked to identify any inconsistencies or repeated information within the dataset.
- A logical relationship between weather variables and flight delays was investigated.
- Cleaning and organizing the data from the CSV file.
- Identified which airlines perform the worst in terms of delays.
- Determined which airlines perform the best.
- Analyzed whether flight performance varies by month, whether a certain airline consistently performs poorly, or if performance fluctuates.
- Identified which routes have the highest probability of falling into the level 1 delay category.

  
  
## Step 6.1 Correlation between Numerical Features

- A logical relationship between weather variables and flight delays was investigated.
  
```Python
numeric_df = df_cleaned.select_dtypes(include=['float64', 'int64'])

delay_columns = ['dep_delay', 'arr_delay']
weather_columns = ['temp', 'dewp', 'humid', 'wind_dir', 'wind_speed', 'wind_gust', 'precip', 'pressure', 'visib']

existing_columns = [col for col in delay_columns + weather_columns if col in numeric_df.columns]
subset_df = numeric_df[existing_columns]

subset_df = subset_df.dropna()

corr = subset_df.corr()

plt.figure(figsize=(12, 10))
plt.matshow(corr, fignum=1, cmap='YlGnBu')
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

for (i, j), val in np.ndenumerate(corr.values):
    plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

plt.title('Correlation between Weather and Delay Time')
plt.show()
```

- Identified which airlines perform the worst in terms of delays.
  
```Python
worst_performer= df_cleaned.groupby("carrier").sum().sort_values(by="dep_delay", ascending=False)
worst_performer.head(5)
```

| carrier | dep_delay | arr_delay | flight   | air_time    | distance | temp     | dewp     | humid       | wind_dir   | wind_speed   | wind_gust    | precip | pressure      | visib      |
|---------|-----------|-----------|----------|-------------|----------|----------|----------|-------------|------------|--------------|--------------|--------|---------------|------------|
| AS      | 271775    | 106750    | 8214162  | 6629773.0   | 52832566 | 1926061.4 | 1625351.1 | 3111531.76 | 6408410.0 | 283662.66688 | 326433.323802 | 224.8616 | 41179032.7    | 359443.60  |
| DL      | 171518    | 14605     | 6610553  | 2830734.0   | 23502339 | 773384.8 | 651917.0 | 1242257.65 | 2594660.0 | 114517.57014 | 131784.529369 | 93.3219 | 16459434.0    | 143012.33  |
| QX      | 121612    | 81767     | 11684962 | 1180098.0   | 6933990  | 941200.3 | 789680.7 | 1482571.91 | 3106660.0 | 135284.54602 | 155682.749873 | 109.6436 | 19840516.4    | 172873.41  |
| WN      | 78754     | 20462     | 3613688  | 770253.0    | 5880952  | 343527.3 | 285479.4 | 519748.79  | 1118510.0 | 46148.57956  | 53106.862388  | 36.2240 | 7081655.7     | 63197.75   |
| AA      | 72289     | 32390     | 2066108  | 725827.0    | 6179705  | 172969.2 | 145300.9 | 274049.26  | 576670.0  | 23971.89818  | 27586.380988  | 21.6781 | 3652974.7     | 32388.15   |

- Determined which airlines perform the best.
  
```Python
best_performer= df_cleaned.groupby("carrier").sum().sort_values(by="dep_delay", ascending=True)
best_performer.head()
```

| carrier | dep_delay | arr_delay | flight  | air_time   | distance | temp    | dewp    | humid     | wind_dir | wind_speed  | wind_gust   | precip | pressure  | visib  |
|---------|-----------|-----------|---------|------------|----------|---------|---------|-----------|----------|-------------|-------------|--------|-----------|--------|
| G4      | 2436      | 2022      | 17123   | 11995.0    | 100247   | 4675.9  | 3926.6  | 6786.27   | 13710.0  | 530.50958   | 610.499814  | 0.7708 | 92620.2   | 862.25 |
| HA      | 5942      | 3335      | 10128   | 241856.0   | 1889778  | 32938.8 | 28690.2 | 58146.26  | 99430.0  | 4072.61042  | 4686.678619 | 4.8977 | 735653.8  | 6459.75|
| F9      | 6300      | 4854      | 172830  | 51015.0    | 401619   | 23002.3 | 18170.0 | 30447.84  | 79200.0  | 3215.27932  | 3700.079136 | 2.8852 | 450716.8  | 4064.06|
| NK      | 8879      | 4393      | 379062  | 89964.0    | 665145   | 39494.5 | 32847.8 | 61641.66  | 129040.0 | 5262.51694  | 6055.999244 | 3.8301 | 835843.0  | 7359.11|
| B6      | 26951     | 19990     | 154254  | 174509.0   | 1542234  | 34901.9 | 26131.1 | 40348.70  | 124520.0 | 5297.04034  | 6095.728083 | 3.6764 | 652737.6  | 5913.37|


- Analyzed whether flight performance varies by month, whether a certain airline consistently performs poorly, or if performance fluctuates.
  
```Python
df['year'] = df_cleaned['date'].dt.year
df['month'] = df_cleaned['date'].dt.month

monthly_dep_delay = df.groupby(['year', 'month'])['dep_delay'].mean().reset_index()

print(monthly_dep_delay)
```

| year | month | dep_delay |
|------|-------|-----------|
| 2022 | 1     | 10.804097 |
| 2022 | 2     | 5.254844  |
| 2022 | 3     | 5.896043  |
| 2022 | 4     | 10.032263 |
| 2022 | 5     | 7.127991  |
| 2022 | 6     | 8.900074  |

```Python
df['year'] = df_cleaned['date'].dt.year
df['month'] = df_cleaned['date'].dt.month

monthly_arr_delay = df.groupby(['year', 'month'])['arr_delay'].mean().reset_index()

print(monthly_arr_delay)
```

| year | month | arr_delay |
|------|-------|-----------|
| 2022 | 1     | 4.648079  |
| 2022 | 2     | -1.099085 |
| 2022 | 3     | 0.134775  |
| 2022 | 4     | 4.611300  |
| 2022 | 5     | 1.708427  |
| 2022 | 6     | 3.790448  |

```Python
avg_delay_by_carrier = df_cleaned.groupby("carrier")["dep_delay"].mean()

plt.figure(figsize=(10, 6))
avg_delay_by_carrier.plot(kind="bar", color="purple", edgecolor="black")
plt.title("Average Departure Delay by Carrier")
plt.xlabel("Carrier")
plt.ylabel("Average Departure Delay (minutes)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
```

![avg_departuredelay_carrier](https://github.com/user-attachments/assets/1c23eafb-9466-4c00-b94f-b829beece06d)


- Identified which routes have the highest probability of falling into the level 1 delay category.
```Python
df['dep_delay'] = df['dep_delay'].apply(lambda x: max(0, x))
df['arr_delay'] = df['arr_delay'].apply(lambda x: max(0, x))

df['dep_delay_cat'] = pd.qcut(df['dep_delay'], 5, labels=False, duplicates='drop')
df['arr_delay_cat'] = pd.qcut(df['arr_delay'], 5, labels=False, duplicates='drop')

max_dep_delay_cat = df['dep_delay_cat'].max()
max_arr_delay_cat = df['arr_delay_cat'].max()

df['dep_delay_level1'] = df['dep_delay_cat'] == max_dep_delay_cat
df['arr_delay_level1'] = df['arr_delay_cat'] == max_arr_delay_cat

df['route'] = df['origin'] + '-' + df['dest']

route_delay_prob = df.groupby('route').agg(
    dep_delay_prob=('dep_delay_level1', 'mean'),
    arr_delay_prob=('arr_delay_level1', 'mean')
).reset_index()

top_dep_delay_routes = route_delay_prob.nlargest(5, 'dep_delay_prob')
top_arr_delay_routes = route_delay_prob.nlargest(5, 'arr_delay_prob')

print("Level 1: Routes with the highest probability of departure delay")
print(top_dep_delay_routes)

print("Level 1: Routes with the highest probability of arrival delay")
print(top_arr_delay_routes)
```


Routes with the Highest Probability of Delay

### Routes with the Highest Probability of Departure Delay
These routes have the highest likelihood of experiencing a delay upon departure.

| Route   | Departure Delay Probability | Arrival Delay Probability |
|---------|-----------------------------|---------------------------|
| PDX-DSM | 0.739                       | 0.696                     |
| PDX-GRR | 0.739                       | 0.696                     |
| PDX-DAL | 0.700                       | 0.700                     |
| PDX-STL | 0.667                       | 0.500                     |
| PDX-IDA | 0.444                       | 0.400                     |

### Routes with the Highest Probability of Arrival Delay
These routes show the highest likelihood of experiencing delays upon arrival.

| Route   | Departure Delay Probability | Arrival Delay Probability |
|---------|-----------------------------|---------------------------|
| PDX-DAL | 0.700                       | 0.700                     |
| PDX-DSM | 0.739                       | 0.696                     |
| PDX-GRR | 0.739                       | 0.696                     |
| PDX-STL | 0.667                       | 0.500                     |
| PDX-IDA | 0.444                       | 0.400                     |

```Python
plt.figure(figsize=(10, 6))
plt.bar(top_dep_delay_routes['route'], top_dep_delay_routes['dep_delay_prob'], color='red')
plt.xlabel('Route')
plt.ylabel('Probability of Departure Delay (Level 1)')
plt.title('Routes with the Highest Probability of Departure Delay (Level 1)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(top_arr_delay_routes['route'], top_arr_delay_routes['arr_delay_prob'], color='blue')
plt.xlabel('Route')
plt.ylabel('Probability of Arrival Delay (Level 1)')
plt.title('Routes with the Highest Probability of Arrival Delay (Level 1)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

```
![routeswith_highestprob_dep_delay](https://github.com/user-attachments/assets/d7559535-a36a-421d-bde8-efcb8ccbd978)

![routeswith_highestprob_arr_delay](https://github.com/user-attachments/assets/9c0e5eeb-61bc-433b-a192-a79e336b31e9)


- This code creates bar charts to visualize the routes with the highest probabilities of experiencing delays, separated by departure and arrival delays.
- First Plot: The code generates a bar chart showing the probability of departure delays for the top routes. The x-axis represents specific routes, while the y-axis shows the probability of delay. The bars are colored red, and the routes are rotated on the x-axis for readability.
- Second Plot: The code produces a similar bar chart for the probability of arrival delays on the same routes. This chart follows the same formatting, but the bars are colored blue to distinguish it from the departure delay chart.

# Step 7: Project Questions

1. Create a categorical variable to understand the relationship between each airline and flight distance.

```Python
bins = [0, 500, 1000, 3000]
labels = ['Short Distance', 'Medium Distance', 'Long Distance']

df_cleaned['distance_category'] = pd.cut(df_cleaned['distance'], bins=bins, labels=labels)
df_cleaned['distance_category']
```



2. Classify flight distances into three main groups: distances under 500 miles, distances between 500-1000 miles, and distances of 1000 miles and above.
   
```Python
bins = [0, 500, 1000, float('inf')]
labels = ['Under 500 miles', '500-1000 miles', 'Above 1000 miles']

df_cleaned['distance_category2'] = pd.cut(df_cleaned['distance'], bins=bins, labels=labels)
df_cleaned['distance_category2']
```

3. Identify the airlines with the highest number of flights conducted.

```Python
max_flyer = df_cleaned.groupby("airline")["distance"].sum().sort_values(ascending=False)
max_flyer
```

Below is the cumulative distance flown by each airline, showing their relative flight activity in miles.

| Airline                     | Total Distance Flown (Miles) |
|-----------------------------|------------------------------|
| Alaska Airlines Inc.        | 52,832,566                   |
| Delta Air Lines Inc.        | 23,502,339                   |
| United Air Lines Inc.       | 8,418,462                    |
| SkyWest Airlines Inc.       | 7,236,636                    |
| Horizon Air                 | 6,933,990                    |
| American Airlines Inc.      | 6,179,705                    |
| Southwest Airlines Co.      | 5,880,952                    |
| Hawaiian Airlines Inc.      | 1,889,778                    |
| JetBlue Airways             | 1,542,234                    |
| Spirit Air Lines            | 665,145                      |
| Frontier Airlines Inc.      | 401,619                      |
| Allegiant Air               | 100,247                      |

This data highlights Alaska Airlines Inc. as the airline with the highest total distance flown, followed by Delta Air Lines Inc. and United Air Lines Inc.


```Python
max_flyer = df_cleaned.groupby("airline")["distance"].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
max_flyer.plot(kind='bar', color='darkgreen')
plt.title('Total Distance Flown by Airlines')
plt.xlabel('Airline')
plt.ylabel('Total Distance (units)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

![totaldistance_flownby_airlines](https://github.com/user-attachments/assets/8061d2ae-4049-4cbe-b477-4276ed399113)


4. Compare the performance of airlines based on delays.

```Python
avg_dep_delay = df_cleaned.groupby("airline")["dep_delay"].mean().reset_index()
avg_dep_delay.columns = ['airline', 'avg_dep_delay']

avg_arr_delay = df_cleaned.groupby("airline")["arr_delay"].mean().reset_index()
avg_arr_delay.columns = ['airline', 'avg_arr_delay']

airline_performance = pd.merge(avg_dep_delay, avg_arr_delay, on='airline')
print(airline_performance)
```

This table shows the average departure and arrival delays (in minutes) for each airline, providing insights into their overall performance.

| Airline                    | Average Departure Delay (mins) | Average Arrival Delay (mins) |
|----------------------------|--------------------------------|------------------------------|
| Alaska Airlines Inc.       | 6.73                           | 2.65                         |
| Allegiant Air              | 26.77                          | 22.22                        |
| American Airlines Inc.     | 20.18                          | 9.04                         |
| Delta Air Lines Inc.       | 10.63                          | 0.91                         |
| Frontier Airlines Inc.     | 14.25                          | 10.98                        |
| Hawaiian Airlines Inc.     | 8.24                           | 4.63                         |
| Horizon Air                | 6.25                           | 4.20                         |
| JetBlue Airways            | 42.11                          | 31.23                        |
| SkyWest Airlines Inc.      | 4.61


```Python
airlines = airline_performance['airline']
avg_dep_delays = airline_performance['avg_dep_delay']
avg_arr_delays = airline_performance['avg_arr_delay']

fig, ax = plt.subplots(figsize=(10, 8))
bar_width = 0.4
index = np.arange(len(airlines))

bar1 = ax.bar(index, avg_dep_delays, bar_width, label='Average Departure Delay', color='skyblue')
bar2 = ax.bar(index + bar_width, avg_arr_delays, bar_width, label='Average Arrival Delay', color='lightgreen')

ax.set_xlabel('Airlines')
ax.set_ylabel('Average Delay (minutes)')
ax.set_title('Average Departure and Arrival Delays by Airline')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(airlines, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.show()
```
![avgdeparture_avgarrivaldelays](https://github.com/user-attachments/assets/887bc781-08c3-4d86-b31d-432035a10e41)


5. Examine whether airline performance varies across different months of the year, and show the top 5 airlines with the best performance and the bottom 5 with the worst performance.

```Python
df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])
df_cleaned['year'] = df_cleaned['date'].dt.year
df_cleaned['month'] = df_cleaned['date'].dt.month

monthly_airline_performance = df_cleaned.groupby(['year', 'month', 'airline'])['dep_delay'].mean().reset_index()
print(monthly_airline_performance)
```
# Monthly Average Departure Delay by Airline (2022)

This table shows the average departure delay for each airline, broken down by month for 2022. This data helps to identify any seasonal trends in delays.

| Year | Month | Airline                   | Average Departure Delay (mins) |
|------|-------|----------------------------|---------------------------------|
| 2022 | 1     | Alaska Airlines Inc.       | 8.84                            |
| 2022 | 1     | Allegiant Air              | 11.88                           |
| 2022 | 1     | American Airlines Inc.     | 19.53                           |
| 2022 | 1     | Delta Air Lines Inc.       | 10.60                           |
| 2022 | 1     | Frontier Airlines Inc.     | 18.14                           |
| ...  | ...   | ...                        | ...                             |
| 2022 | 6     | JetBlue Airways            | 66.88                           |
| 2022 | 6     | SkyWest Airlines Inc.      | 3.79                            |
| 2022 | 6     | Southwest Airlines Co.     | 13.81                           |
| 2022 | 6     | Spirit Air Lines           | 18.12                           |
| 2022 | 6     | United Air Lines Inc.      | 7.95                            |

_Note: Only a subset of the data is shown here. The complete data includes monthly average departure delays for all airlines from January to June 2022._

```Python
monthly_airline_performance = monthly_airline_performance.sort_values(by=['year', 'month'])
plt.figure(figsize=(12, 8))
for airline in monthly_airline_performance['airline'].unique():
    data = monthly_airline_performance[monthly_airline_performance['airline'] == airline]
    plt.plot(data['year'].astype(str) + '-' + data['month'].astype(str).str.zfill(2), data['dep_delay'], marker='o', label=airline)

plt.xlabel('Year-Month')
plt.ylabel('Average Departure Delay (minutes)')
plt.title('Monthly Average Departure Delay by Airline')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```






```Python
df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])
df_cleaned['year'] = df_cleaned['date'].dt.year
df_cleaned['month'] = df_cleaned['date'].dt.month

monthly_airline_performance2 = df_cleaned.groupby(['year', 'month', 'airline'])['arr_delay'].mean().reset_index()
print(monthly_airline_performance2)
```

# Monthly Average Arrival Delay by Airline (2022)

This table presents the average arrival delay for each airline, broken down by month for 2022. This data helps in identifying arrival delay trends across different months and airlines.

| Year | Month | Airline                   | Average Arrival Delay (mins) |
|------|-------|----------------------------|------------------------------|
| 2022 | 1     | Alaska Airlines Inc.       | 4.13                         |
| 2022 | 1     | Allegiant Air              | 8.75                         |
| 2022 | 1     | American Airlines Inc.     | 5.20                         |
| 2022 | 1     | Delta Air Lines Inc.       | 0.14                         |
| 2022 | 1     | Frontier Airlines Inc.     | 13.82                        |
| ...  | ...   | ...                        | ...                          |
| 2022 | 6     | JetBlue Airways            | 56.77                        |
| 2022 | 6     | SkyWest Airlines Inc.      | -0.48                        |
| 2022 | 6     | Southwest Airlines Co.     | 5.93                         |
| 2022 | 6     | Spirit Air Lines           | 13.97                        |
| 2022 | 6     | United Air Lines Inc.      | -3.73                        |

_Note: Only a subset of the data is shown here. The complete dataset includes monthly average arrival delays for all airlines from January to June 2022._

This breakdown enables an analysis of arrival delay patterns for each airline, providing insight into months where airlines perform better or worse in terms of punctuality.


```Python
monthly_airline_performance2 = monthly_airline_performance2.sort_values(by=['year', 'month'])
plt.figure(figsize=(12, 8))
for airline in monthly_airline_performance2['airline'].unique():
    data = monthly_airline_performance2[monthly_airline_performance2['airline'] == airline]
    plt.plot(data['year'].astype(str) + '-' + data['month'].astype(str).str.zfill(2), data['arr_delay'], marker='o', label=airline)
    
plt.xlabel('Year-Month')
plt.ylabel('Average Arrival Delay (minutes)')
plt.title('Monthly Average Arrival Delay by Airline')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```


``` Python
total_performance = df_cleaned.groupby("airline")["distance"].sum().reset_index()
best_5_airlines = total_performance.sort_values(by="distance", ascending=False).head(5)
print(best_5_airlines)
```
# Airline Distance Analysis

This table shows the total distance flown by each airline, providing an indication of their flight activity and network reach.

| Airline                 | Total Distance Flown (Miles) |
|-------------------------|------------------------------|
| Alaska Airlines Inc.    | 52,832,566                   |
| Delta Air Lines Inc.    | 23,502,339                   |
| United Air Lines Inc.   | 8,418,462                    |
| SkyWest Airlines Inc.   | 7,236,636                    |
| Horizon Air             | 6,933,990                    |

This data highlights Alaska Airlines Inc. as the airline with the highest total distance flown, followed by Delta Air Lines Inc. and United Air Lines Inc.

``` Python
otal_performance = df_cleaned.groupby("airline")["distance"].sum().reset_index()
best_5_airlines = total_performance.sort_values(by="distance", ascending=False).head(5)

plt.figure(figsize=(10, 6))
plt.bar(best_5_airlines['airline'], best_5_airlines['distance'], color='skyblue')
plt.xlabel('Airline')
plt.ylabel('Total Distance Flown')
plt.title('Top 5 Airlines by Total Distance Flown')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

```
![top5_airlines_distanceflown](https://github.com/user-attachments/assets/4e58a3f4-3284-4542-ba63-5409e35e7527)


``` Python
worst_5_airlines = total_performance.sort_values(by="distance").head(5)
print(worst_5_airlines)
```
# Airline Distance Analysis (Continued)

This table shows additional airlines with their respective total distance flown, giving further insights into their operational scale.

| Airline                   | Total Distance Flown (Miles) |
|---------------------------|------------------------------|
| Allegiant Air             | 100,247                      |
| Frontier Airlines Inc.    | 401,619                      |
| Spirit Air Lines          | 665,145                      |
| JetBlue Airways           | 1,542,234                    |
| Hawaiian Airlines Inc.    | 1,889,778                    |

This data highlights lower total distances flown by these airlines, reflecting smaller networks or more regionally focused operations compared to larger carriers.

``` Python
worst_5_airlines = total_performance.sort_values(by="distance").head(5)
plt.figure(figsize=(10, 6))
plt.barh(worst_5_airlines['airline'], worst_5_airlines['distance'], color='lightcoral')
plt.xlabel('Total Distance Flown')
plt.ylabel('Airline')
plt.title('Bottom 5 Airlines by Total Distance Flown')
plt.tight_layout()
plt.show()
```

![Bottom5Airlines_distanceflown](https://github.com/user-attachments/assets/a0402ecb-fc7d-4ba0-82ee-a2b252942f05)



6. Determine the percentage of flight cancellations and display how this varies by airline.

``` Python
df_copy['cancelled'] = df_copy['dep_time'].isna() | df_copy['arr_time'].isna()
total_flights = len(df_copy)
cancelled_flights_count = df_copy['cancelled'].sum()
cancelled_flights_percentage = (cancelled_flights_count / total_flights) * 100

print(f"Total number of flights: {total_flights}")
print(f"Number of cancelled flights: {cancelled_flights_count}")
print(f"Percentage of cancelled flights: {cancelled_flights_percentage:.2f}%")
airline_cancelled_percentage = df_copy.groupby('airline')['cancelled'].mean() * 100

print("\nPercentage of cancelled flights by airline:")
print(airline_cancelled_percentage)
```
# Flight Cancellation Analysis

### Overall Cancellation Statistics
- **Total Number of Flights**: 111,006
- **Number of Cancelled Flights**: 2,537
- **Percentage of Cancelled Flights**: 2.29%

### Percentage of Cancelled Flights by Airline

| Airline                    | Percentage of Cancelled Flights (%) |
|----------------------------|-------------------------------------|
| Alaska Airlines Inc.       | 3.21                               |
| Allegiant Air              | 6.19                               |
| American Airlines Inc.     | 2.32                               |
| Delta Air Lines Inc.       | 2.32                               |
| Frontier Airlines Inc.     | 2.64                               |
| Hawaiian Airlines Inc.     | 0.55                               |
| Horizon Air                | 1.42                               |
| JetBlue Airways            | 4.76                               |
| SkyWest Airlines Inc.      | 1.30                               |
| Southwest Airlines Co.     | 1.42                               |
| Spirit Air Lines           | 4.43                               |
| United Air Lines Inc.      | 1.34                               |

This analysis shows that **Allegiant Air** has the highest cancellation rate at 6.19%, while **Hawaiian Airlines Inc.** has the lowest at 0.55%. These percentages provide insight into airline reliability in terms of flight cancellations.

``` Python
df_copy['cancelled'] = df_copy['dep_time'].isna() | df_copy['arr_time'].isna()
total_flights = len(df_copy)
cancelled_flights_count = df_copy['cancelled'].sum()
cancelled_flights_percentage = (cancelled_flights_count / total_flights) * 100

airline_cancelled_percentage = df_copy.groupby('airline')['cancelled'].mean() * 100
plt.figure(figsize=(12, 8))
bars = plt.bar(airline_cancelled_percentage.index, airline_cancelled_percentage.values, color='blue', edgecolor='black')

plt.xlabel('Airline', fontsize=12, fontweight='bold')
plt.ylabel('Percentage of Cancelled Flights', fontsize=12, fontweight='bold')
plt.title('Percentage of Cancelled Flights by Airline', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{bar.get_height():.1f}%', 
             ha='center', va='bottom', fontsize=9)
plt.show()
```
![PercentageofCancelledFlights](https://github.com/user-attachments/assets/8e378408-c7a6-4740-a6e0-03cea08253d6)

7. Is there a specific time of day and/or time of year with higher delay durations?
8. Which routes have the highest delay durations?




### Results/Findings 
The analysis results are summarised as follows:
1. 
2. 
3. 


### Recommendations
Based on the analysis, we recommend the following actions:


### Limitations


### References
