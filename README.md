# Analysis-of-Flight-Delays-and-Cancellations

## Table Of Contents

- [Project Overview](#project-overview)
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

### Project Overview
The objective of this project is to analyze a comprehensive dataset containing flight information, including departure and arrival times, delays, distances, weather conditions, and airline details. By exploring this data, we aim to identify significant patterns and correlations that contribute to flight delays. This analysis will help to understand the key factors influencing delays, allowing us to develop predictive models and actionable insights to improve airline punctuality and operational efficiency. Through careful data exploration and modeling, this project seeks to support better decision-making for airlines and enhance the overall passenger experience by reducing delays.

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

- Which airlines show the worst/best performance in terms of delays?
- Does flight performance vary by the month of the year?
- Which routes are most likely to fall into the 1st level delay category?

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


# Step 3.2 Summary Statistics for Numerical Variables

```
df.describe().T
```








#### Exploratory Data Analysis
```Python
df.info()
df.isnull().sum()
df.describe().T
```
#### Project Details
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

























### Results/Findings 
The analysis results are summarised as follows:
1. 
2. 
3. 


### Recommendations
Based on the analysis, we recommend the following actions:


### Limitations


### References
