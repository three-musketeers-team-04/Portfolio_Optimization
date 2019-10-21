###########################
# PORTFOLIO OPTIMIZER
###########################
# Customer input: Inputs upto 5 stock tickers
# Portfolio Optimizer performs the below tasks:
# Ingests historical data upto 10 years
# Processes and manipulates data
# Stores it into a database (TBD)
# Calculates different statistics, returns, and risk
# Selects stocks to keep in portfolio based on various variables (TBD)
# Run a Monte Carlo Simulation with random weights to pick best portfolio return
# Visualizes an optimized porfolio based on risk-return tradeoff
# Built in trailing stop for certain thresholds (TBD)
# Run a Monte Carlo Simulation for returns of each stock 
# Calculate confidence intervals for 6-12 month prediction

################
# KNOWLEDGE REPO
################
# How do we accomodate for variations in allocations across the portfolio?
# How does the investment amount come into play?
# Do we need to run a monte carlo for forecasting and optimization seperately?
# What are the other benefits of OOP besides organization
# How do we incorporate the risk profile? Low Risk [20/80], High Risk [60/40] - US treasury 10 year bond
# Are we running the efficient frontier with the bond allocation stocks? Data needed? TBD
# what is the output of the efficient frontier? Allocation and visual?

####################
# LIBRARY IMPORTS 
#################
import pandas as pd
import numpy as np
import sklearn as sk   

import tkinter as tk
import tkinter.ttk as ttk

from collections import OrderedDict
from PIL import ImageTk, Image
from threading import Thread
import time
import datetime

import user_interface
import pandas_datareader as pdr
# import ticker_symbols


#################
# DATA INGESTION
#################
# class Ingestion():
#     """Extracts data from an external source into a python dataframe"""
#     def __init__(self):
#         pass

#     def extract_data(ticker_symbols_list):
#         """Creates an api connection with yahoo finance via pandas datareader"""
#         start_sp = datetime.datetime(1999, 10, 1)
#         end_sp = datetime.datetime(2019, 10, 1)
        
#         ticker_symbols_list.extend('SPY', '^TNX')
#         main_df = pdr.get_data_yahoo(ticker_symbols_list, start_sp, end_sp)['Adj Close']
#         return main_df    

app = user_interface.UI()
app.mainloop()

# ###################
# # DATA MANIPULATION 
# ###################
# class Manipulation(self):
#     def __init__(self):
#         pass 

#     def clean_nulls(main_df):
#         """For stocks that do not have the max week impute with mean so everytihing is similar"""
#         final_df = main_df.fillna(main_df.mean())
#         return final_df
       
#     # def apply_pivot_function():
#     #     """Pivot the dataframe for analysis"""
#     #     final_df = normalized_df.unstack()
#     #     return final_df

# ##########################
# # LOADING HISTORICAL DATA
# #########################
# # class LoadingDatabase(self):
# #     def create_connection_string():
# #         pass 
    
# #     def create_cursor():
# #         pass 
    
# #     def upload_data():
# #         pass 


# ########################
# # DESCRIPTIVE STATISTICS
# ########################
# class Reporting(self):
#     def __init__(self):
#         pass
        
#     def descriptive_stats(final_df):
#         """Stats around returns, standard deviation, mean returns, and covariance"""    
#         returns_df = final_df.pct_change() 
        
    
#         mean_returns_series = returns_df.mean()
#         mean_returns_series = mean_returns_df.round(4)*1000

#         std_dev_series = returns_df.std()
#         std_dev_series = std_dev_series.round(4)*1000

#         # Loop through the columns to get mean returns and std dev for each stock variable
#         # stock_mean_returns = mean_returns['stock']
#         return cov_df 
    
#     def beta(returns_df):
#         """Calculate covariance, variances, and beta values"""
#         cov_df = returns_df.cov()  # Need to specify which two stocks are we creating a covariance for to understand Beta 
        
#         var_series = returns_df.var()
#         var_series = std_dev_series.round(4)*1000

# #         merged_df['rolling_covariance'] = merged_df['BERKSHIRE HATHAWAY INC'].rolling(window = 60).cov(merged_df['S&P 500'])
# #         merged_df['rolling_variance'] = merged_df['S&P 500'].rolling(window = 60).var()
# #         merged_df['rolling_beta'] = merged_df['rolling_covariance']/merged_df['rolling_variance']
# # ​
# #         merged_df['rolling_beta'].plot(figsize = (25,10), title = 'Berkshire Hathaway Inc. Beta')
         
#         pass   


# ###################################################
# # OPTIMAL PORTFOLIO SELECTION AND ASSET FORECASTING
# ##################################################
# class Analysis(self):
#     def __init__(self):
#         pass

# # Optimal Portfolio
      
# # Forecasting returns for each stock within the portfolio
#     def monte_carlo_simulation():
#         # Set number of simulations and trading days
#         num_simulations = 1000
#         num_trading_days = 252 * 3

#         # Set last closing prices of `TSLA` and `SPHD`
#         tsla_last_price = df['TSLA']['close'][-1]
#         sphd_last_price = df['SPHD']['close'][-1]

#         # Initialize empty DataFrame to hold simulated prices for each simulation
#         simulated_price_df = pd.DataFrame()
#         portfolio_cumulative_returns = pd.DataFrame()

#         # Run the simulation of projecting stock prices for the next trading year, `1000` times
#         for n in range(num_simulations):

#             # Initialize the simulated prices list with the last closing price of `TSLA` and `SPHD`
#             simulated_tsla_prices = [tsla_last_price]
#             simulated_sphd_prices = [sphd_last_price]
            
#             # Simulate the returns for 252 * 3 days
#             for i in range(num_trading_days):
                
#                 # Calculate the simulated price using the last price within the list
#                 simulated_tsla_price = simulated_tsla_prices[-1] * (1 + np.random.normal(avg_daily_return_tsla, std_dev_daily_return_tsla))
#                 simulated_sphd_price = simulated_sphd_prices[-1] * (1 + np.random.normal(avg_daily_return_sphd, std_dev_daily_return_sphd))
                
#                 # Append the simulated price to the list
#                 simulated_tsla_prices.append(simulated_tsla_price)
#                 simulated_sphd_prices.append(simulated_sphd_price)
            
#             # Append a simulated prices of each simulation to DataFrame
#             simulated_price_df["TSLA prices"] = pd.Series(simulated_tsla_prices)
#             simulated_price_df["SPHD prices"] = pd.Series(simulated_sphd_prices)
            
#             # Calculate the daily returns of simulated prices
#             simulated_daily_returns = simulated_price_df.pct_change()
            
#             # Set the portfolio weights (75% TSLA; 25% SPHD)
#             weights = [0.25, 0.75]

#             # Use the `dot` function with the weights to multiply weights with each column's simulated daily returns
#             portfolio_daily_returns = simulated_daily_returns.dot(weights)
            
#             # Calculate the normalized, cumulative return series
#             portfolio_cumulative_returns[n] = (1 + portfolio_daily_returns.fillna(0)).cumprod()

#         # Print records from the DataFrame
#         portfolio_cumulative_returns.head()
#         ending_cumulative_returns = portfolio_cumulative_returns.iloc[-1, :]
#         return portfolio_cumulative_returns, ending_cumulative_returns

#     def probability_distribution(ending_cumulative_returns):
#         ending_cumulative_returns.value_counts(bins=10) / len(ending_cumulative_returns)
#         cumulative_returns = (1 + portfolio_returns).cumprod() - 1
#         cumulative_returns.head()

#     def calculate_confidence_intervals():
#         """Calculate confidence intervals for return"""
#         confidence_interval = ending_cumulative_returns.quantile(q=[0.025, 0.975])
#         # Set initial investment
#         initial_investment = 10000

#         # Calculate investment profit/loss of lower and upper bound cumulative portfolio returns
#         investment_pnl_lower_bound = initial_investment * confidence_interval.iloc[0]
#         investment_pnl_upper_bound = initial_investment * confidence_interval.iloc[1]
                                                    
# # Print the results
#     # print(f"There is a 95% chance that an initial investment of $10,000 in the portfolio"
#     #   f" over the next 252 * 3 trading days will end within in the range of"
#     #   f" ${investment_pnl_lower_bound} and ${investment_pnl_upper_bound}")
#     #     return confidence_interval
    
    
#     def calc_portfolio_returns_compared_to_benchmark():
#         """For a specific investment calculate cumulative portfolio returns to the sp_500"""
#         pass 
      
# # Efficient frontier    
#     def efficient_frontier():
#         """Create efficient frontier based on monte carlo simulation to pick best portfolio mix"""
#         pass

#     def monte_carlo_returns():
#         """Run monte-carlo simulation of each stock return"""
#         pass

#     def portfolio_returns():
#         """Add weights to each stock within the portfolio and calculate returns"""
#         pass

#     def monte_carlo_weights():
#         """Run portfolio returns with random weights"""
#         pass

#     def calculate_ratios():
#         """Calculate all 8 ratios for the final chart"""
#         pass 
    

# #########################
# # DASHBOARD VISUALIZATION
# #########################
# class Visualization(self):
#     """Create visualizations of all analysis and pack into a dashboard"""

#     def plot_monte_carlo():
#         """Plotting all plots from the monte carlo analysis"""
#         plot_title = f"{n+1} Simulations of Cumulative Portfolio Return Trajectories Over the Next 252 Trading Days"
#         portfolio_cumulative_returns.plot(legend=None, title=plot_title)
            
#     def plot_freq_dist_of_last_day():
#         ending_cumulative_returns.plot(kind='hist', bins=10)


#     def prob_dist_at_certain_confidence_interval():
#         plt.figure();
#         ending_cumulative_returns.plot(kind='hist', density=True, bins=10)
#         plt.axvline(confidence_interval.iloc[0], color='r')
#         plt.axvline(confidence_interval.iloc[1], color='r')

#     def exponential_weighted_average():
#         merged_subset.ewm(halflife=21, adjust = True).mean().plot(figsize = (25,10), title = 'Exponetially Weighted Average')
    
#     def plot_beta():
#         merged_df['rolling_covariance'] = merged_df['BERKSHIRE HATHAWAY INC'].rolling(window = 60).cov(merged_df['S&P 500'])
#         merged_df['rolling_variance'] = merged_df['S&P 500'].rolling(window = 60).var()
#         merged_df['rolling_beta'] = merged_df['rolling_covariance']/merged_df['rolling_variance']

#         merged_df['rolling_beta'].plot(figsize = (25,10), title = 'Berkshire Hathaway Inc. Beta')

#     def plot_rolling_standard_deviation():
#         final_df.rolling(window=21).std().plot(figsize = (25,10))
    
#     def plot_to_show_risk():
#         bplot = merged_df.boxplot(figsize = (25,10))
#         bplot.axes.set_title("Portfolio Risk",
#                     fontsize=12)

#     def plot_cum_returns():
#         cum_returns_df = (1 + merged_df).cumprod()
#         cum_returns_df.plot(subplots = False, figsize = (25,10))

    
#     def plot_daily_returns():
#         merged_df.plot(subplots = False, figsize = (25,10))
    
    
#     def plot_future_returns_prediction():
#         """Plotting prediction for 6 months to a year for each stock"""
#         pass

#     def plot_scatter_matrix():
#         """Creating a scatter matrix of stocks with histograms for distributions"""
#         pass 

#     def plot_heat_map():
#         """Create a heat map for the correlation matrix"""
#         pass 

#     def plot_efficient_frontier():
#         """Plot efficient frontier with the highest Sharpe and Sortino Ratio"""
#         pass 

#     def create_dashboard():
#         """Pack all the visualizations within a dashboard """
#         pass

# ##################################
# # CONNECTING OUTPUT TO A FRONT-END 
# ##################################
# class Visualization(self):
#     def __init__():
#         pass
    
    

# ###############################
# # ENTIRE PROCESS LOGIC SEQUENCE
# ###############################
# # def main():
# #     """Logic for entire process"""
# #     main_df = extract_data(ticker_symbols_list)
# #     maind_df.head()

# # if __name__ == '__main__':
# #     main() 

