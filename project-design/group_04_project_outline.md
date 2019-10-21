# TITLE: Portfolio optimization

# TEAM MEMBERS: Armen, Min, and Roland

# PROJECT OUTLINE

## Problem statement 
* Optimize portfolio based on inputs by investors.

## Potential Data sources
* Yahoo Finance, Quandl, Quantopian, and IEX Finance

## Pipeline Design
### FRONT END DESIGN
* Create main windown
* Create objects (e.g. frames, buttons, icons) within the main window
* Create functionality for event handling
* Outcomes: Capture Ticker symbols in a list, investor Profile to allocate assets during optimization and investment amount.

### DATA INGESTION (Skills: APIs and Python)
* Based on input extract stock prices for the past 10 years

### DATA PROCESSING AND MANIPULATION (Skills: Python Pandas)
* Clean and manipulate stock data

### LOADING DATA INTO A DATABASE (Skills: PostgresSQL, SQL, Python) 
* Push data to a PostgresSQL database (TBD)

### DESCRIPTIVE STATISTICS (Skills: Python Quant)
* Explore mean, std deviation, variance, covariance, rolling mean, rolling std-dev (7-21-180)
* Determine risk, and return of portfolios
* Run Correlation analysis

### ANALYSIS (Skills: Python Quant)
* Run various optimization functions (E.g. Maximize Returns, Minimize Volatility, Maximize Sharpe Ratio, Minimize Variance, Maximize Information Ratio, Minimize CvAR,      Maximize Sortinos Ratio, Risk Parity)
* Randomize weights  
* Eliminate stocks from the portfolio with highest volatility and also stocks that are higly correlated
* Run Monte Carlo simulations to get the probability of the objective functions over various outcomes (E.g.returns, closing prices, weights)
* TBD: Precicting stock prices based on closing price and other factors (e.g. Machine learning techniques like Regression, K-nearest neighbors)

### VISUALIZATIONS (Skills: PyViz, Plotly, HvPlot, Panel)
* Plotting the returns, comparisons and optimized portfolios
* Create a scatter matrix
* Plotting correlation matrix in form of a heat map
* Monte Carlo Simulations: Creating interactive pltots with a drop down of stocks
* Forecasted trend: Creating an ineractive plot with a drop down of stocks
* Incorporte everything in a dashboard or a report for investors to peruse


