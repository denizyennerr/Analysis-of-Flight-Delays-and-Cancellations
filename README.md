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
This project analyzes factors influencing flight delays and cancellations by examining flight data, including departure and arrival times, weather conditions, and airline performance. The goal is to identify patterns and correlations to improve the aviation industry's operational efficiency and passenger satisfaction.

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

We start with importing the necessary libraries.

```Python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
```
Then 
```Python
df=pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\Uçuş Rötarları ve İptallerinin Analizi.csv")
df = pd.DataFrame(df)
df.head()
```
```Python
df_copy= pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\Uçuş Rötarları ve İptallerinin Analizi.csv")
df_copy
```
Exploratory Data Analysis
```Python
df.info()
df.isnull().sum()
df.describe().T
```
Project Details
-- A date-time index was created for Time Series Analysis and Forecasting, allowing for more effective plotting and time series evaluations. 
-- Departure and arrival times were corrected to ensure the accuracy of the data.
-- Duplicates were checked to identify any inconsistencies or repeated information within the dataset.
-- A logical relationship between weather variables and flight delays was investigated. 
-- Cleaning and organizing the data from the CSV file.
-- Identified which airlines perform the worst in terms of delays.
-- Determined which airlines perform the best.
-- Analyzed whether flight performance varies by month, whether a certain airline consistently performs poorly, or if performance fluctuates.
-- Identified which routes have the highest probability of falling into the level 1 delay category.


### Results/Findings 
The analysis results are summarised as follows:
1. 
2. 
3. 


### Recommendations
Based on the analysis, we recommend the following actions:


### Limitations


### References
