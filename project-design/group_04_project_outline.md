# TITLE: Predicting stock prices


# TEAM MEMBERS: Armen, Min, and Roland


# PROJECT OUTLINE


# Problem statement 
* Predict 6-month price movements of tech stocks (APPL, GOOG, FB, MSFT, AMZ)        based on historical prices.

# Potential Data sources
* IEX Finance, Yahoo Finance, Quandl, Quantopian, Kaggle, US fundamental archives

# Pipeline Design
### DATA INGESTION (Skills: APIs and Python)
* Extract tech stock prices for the past 10 years

### DATA PROCESSING AND MANIPULATION (Skills: Python Pandas)
* Clean and manipulate stock data

### LOADING DATA INTO A DATABASE (Skills: PostgresSQL, SQL, Python) 
* Push data to a PostgresSQL database

### DESCRIPTIVE STATISTICS (Skills: Python Quant)
* Explore rolling mean, moving averages, and return rate of stocks
* Determine risk and return (standard deviation)
* Correlation analysis - Does one competitor affect others

### ANALYSIS (Skills: Python Quant)
* Analyze competitor stocks effect on each other
* Run Monte Carlo simulations
* Precicting stock prices based on closing price and other factors
    - Testing various models (Linear, log-linear, knn)
    - Running cross-validations
    - Predicting

### VISUALIZATIONS (Skills: PyViz, Plotly, HvPlot, Panel)
* Plotting the prediction
* Create a scatter matrix
* Plotting correlation matrix in form of a heat map
* Monte Carlo Simulations: Creating interactive pltots with a drop down of stocks
* Forecasted trend: Creating an ineractive plot with a drop down of stocks
* Incorporte everything in a dashboard


