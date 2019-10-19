#################
# LIBRARY IMPORTS 
#################
import pandas as pd
import numpy as np
import sklearn as sk   
import time
import datetime
import pandas_datareader as pdr
import subprocess
# import tabulate
import matplotlib.pyplot as plt

import tkinter as tk
import tkinter.ttk as ttk
from collections import OrderedDict
from PIL import ImageTk, Image
from threading import Thread

import user_interface

def extract_data(ticker_symbols_list):
    """Creates an api connection with yahoo finance via pandas datareader"""
    start_sp = datetime.datetime(2018, 10, 1)
    end_sp = datetime.datetime(2019, 10, 1)
    
    ticker_symbols_list = [symb for symb in ticker_symbols_list if symb != ""]

    # ticker_symbols_list.extend(['SPY', '^TNX'])
    main_df = pdr.get_data_yahoo(ticker_symbols_list, start_sp, end_sp)['Close']
    print(main_df.head())
    # manipulate = Manipulation()
    # final_df = manipulate.clean_nulls(main_df)
    return main_df, ticker_symbols_list


def clean_nulls(main_df):
    """For stocks that do not have the max week impute with mean so everytihing is similar"""
    final_df = main_df.fillna(main_df.mean())
    return final_df


def descriptive_stats(final_df):
    """Stats around returns, standard deviation, mean returns, and covariance"""    
    returns_df = final_df.pct_change() 
    
    mean_returns_series = returns_df.mean()
    std_dev_series = returns_df.std()

    # Loop through the columns to get mean returns and std dev for each stock variable
    # stock_mean_returns = mean_returns['stock']
    return mean_returns_series, std_dev_series 

def beta(returns_df,std_dev_series):
    """Calculate covariance, variances, and beta values"""
    cov_df = returns_df.cov()  # Need to specify which two stocks are we creating a covariance for to understand Beta 
    
    var_series = returns_df.var()
    var_series = std_dev_series.round(4)*1000

    beta = var_series/cov_df

#         merged_df['rolling_covariance'] = merged_df['BERKSHIRE HATHAWAY INC'].rolling(window = 60).cov(merged_df['S&P 500'])
#         merged_df['rolling_variance'] = merged_df['S&P 500'].rolling(window = 60).var()
#         merged_df['rolling_beta'] = merged_df['rolling_covariance']/merged_df['rolling_variance']
# ​
#         merged_df['rolling_beta'].plot(figsize = (25,10), title = 'Berkshire Hathaway Inc. Beta')
        
    pass   

# Forecasting returns for each stock within the portfolio
def monte_carlo_simulation(final_df, ticker_symbols_list, avg_daily_returns, std_dev_daily_returns, weights, num_trading_days=3):
    # Set number of simulations and trading days
    num_simulations = 1000
    num_trading_days = 252 * num_trading_days

    # Set last closing prices of `TSLA` and `SPHD`
    # tsla_last_price = df['TSLA']['close'][-1]
    # sphd_last_price = df['SPHD']['close'][-1]
    last_prices = final_df.iloc[-1,:]

    avg_daily_returns_list = avg_daily_returns.to_list()
    std_dev_daily_returns_list = std_dev_daily_returns.to_list()

    # Initialize empty DataFrame to hold simulated prices for each simulation
    simulated_price_df = pd.DataFrame()
    portfolio_cumulative_returns = pd.DataFrame()

    # Run the simulation of projecting stock prices for the next trading year, `1000` times
    for n in range(num_simulations):

        # Initialize the simulated prices list with the last closing price of `TSLA` and `SPHD`
        simulated_prices = [[lp] for lp in last_prices]
        
        # Simulate the returns for 252 * 3 days
        for i in range(num_trading_days):
            
            # Calculate the simulated price using the last price within the list
            simulated_last_prices = [simulated_prices[i][-1] * (1 + np.random.normal(avg_daily_returns_list[i], std_dev_daily_returns_list[i])) for i in range(len(avg_daily_returns_list))]

            # Append the simulated price to the list
            for i in range(len(simulated_prices)):
                simulated_prices[i].append(simulated_last_prices[i])

        
        # Append a simulated prices of each simulation to DataFrame

        for i in range(len(ticker_symbols_list)):
            simulated_price_df[ticker_symbols_list[i] + " prices"] = pd.Series(simulated_prices[i])
        
        # Calculate the daily returns of simulated prices
        simulated_daily_returns = simulated_price_df.pct_change()

        # Use the `dot` function with the weights to multiply weights with each column's simulated daily returns
        portfolio_daily_returns = simulated_daily_returns.dot(weights)
        
        # Calculate the normalized, cumulative return series
        portfolio_cumulative_returns[n] = (1 + portfolio_daily_returns.fillna(0)).cumprod()

    # Print records from the DataFrame
    portfolio_cumulative_returns.head()
    ending_cumulative_returns = portfolio_cumulative_returns.iloc[-1, :]
    return portfolio_cumulative_returns, ending_cumulative_returns

def probability_distribution(ending_cumulative_returns):
    ending_cumulative_returns.value_counts(bins=10) / len(ending_cumulative_returns)
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    cumulative_returns.head()

def calculate_confidence_intervals():
    """Calculate confidence intervals for return"""
    confidence_interval = ending_cumulative_returns.quantile(q=[0.025, 0.975])
    # Set initial investment
    initial_investment = 10000

    # Calculate investment profit/loss of lower and upper bound cumulative portfolio returns
    investment_pnl_lower_bound = initial_investment * confidence_interval.iloc[0]
    investment_pnl_upper_bound = initial_investment * confidence_interval.iloc[1]
                                                    
# Print the results
    # print(f"There is a 95% chance that an initial investment of $10,000 in the portfolio"
    #   f" over the next 252 * 3 trading days will end within in the range of"
    #   f" ${investment_pnl_lower_bound} and ${investment_pnl_upper_bound}")
    #     return confidence_interval
    
    
def calc_portfolio_returns_compared_to_benchmark():
    """For a specific investment calculate cumulative portfolio returns to the sp_500"""
    pass 


def efficient_frontier():
    pass


# Visualization
def plot_data_table(final_df):
    plt.clf()
    final_df.round(4)*1000
    final_df.head().to_html('images/table.html')
    table = subprocess.call('wkhtmltoimage -f png --width 0 table.html table.png', shell=True)
    return table


def plot_returns(final_df):
    plt.clf()
    final_df.plot(subplots = False, figsize = (25,10))
    # fct()
    # ax=plt.gca() #get the current axes
    # PCM=ax.get_children()[2] #get the mappable, the 1st and the 2nd are the x and y axes
    # plt.figure(figsize=(20,10))
    # plt.grid(True)
    # plt.xlabel('Expected Volatility')
    # plt.ylabel('Expected Return')
    # plt.colorbar(label='Sharpe Ratio')
    # plt.title('Portfolios of Many Assets')
    # plt.tight_layout()
    plt.savefig('images/portfolio_compositions', dpi=100)
    # plt.imshow()


def plot_monte_carlo(portfolio_cumulative_returns):
        """Plotting all plots from the monte carlo analysis"""
        plt.clf()
        n = 10
        plot_title = f"{n+1} Simulations of Cumulative Portfolio Return Trajectories Over the Next 252 Trading Days"
        portfolio_cumulative_returns.plot(legend=None, title=plot_title)
        plt.savefig('images/montecarlo_simulation', dpi=100)


def plot_freq_dist_of_last_day(ending_cumulative_returns):
    plt.clf()
    ending_cumulative_returns.plot(kind='hist', bins=10, title="Ending Cumulative Returns")
    plt.savefig('images/ending_cumulative_returns', dpi=100)


def prob_dist_at_certain_confidence_interval(ending_cumulative_returns):
    plt.clf()
    confidence_interval = ending_cumulative_returns.quantile(q=[0.025, 0.975])
    ending_cumulative_returns.plot(kind='hist', density=True, bins=10)
    plt.axvline(confidence_interval.iloc[0], color='r')
    plt.axvline(confidence_interval.iloc[1], color='r')
    plt.savefig('images/ending_cumulative_returns_with_confidence_interval', dpi=100)


def plot_exponential_weighted_average(final_df):
    plt.clf()
    final_df.ewm(halflife=21, adjust = True).mean().plot(figsize = (25,10), title = 'Exponetially Weighted Average')
    plt.savefig('images/exponentially_weighted_average')


# def plot_beta():
    # merged_df['rolling_covariance'] = merged_df['BERKSHIRE HATHAWAY INC'].rolling(window = 60).cov(merged_df['S&P 500'])
    # merged_df['rolling_variance'] = merged_df['S&P 500'].rolling(window = 60).var()
    # merged_df['rolling_beta'] = merged_df['rolling_covariance']/merged_df['rolling_variance']
    # merged_df['rolling_beta'].plot(figsize = (25,10), title = 'Berkshire Hathaway Inc. Beta')


def plot_rolling_standard_deviation(final_df):
    plt.clf()
    final_df.rolling(window=21).std().plot(figsize = (25,10))
    plt.savefig('images/rolling_standard_deviation')


def plot_to_show_risk(final_df):
    plt.clf()
    bplot = final_df.boxplot(figsize = (25,10))
    bplot.axes.set_title("Portfolio Risk", fontsize=12)
    plt.savefig('images/box_plot_for_risk')


# def plot_cum_returns():
    # cum_returns_df = (1 + merged_df).cumprod()
    # cum_returns_df.plot(subplots = False, figsize = (25,10))


def plot_daily_returns(final_df):
    plt.clf()
    final_df.plot(subplots = False, figsize = (25,10))
    plt.savefig('images/daily_returns')


# def plot_future_returns_prediction():
    # """Plotting prediction for 6 months to a year for each stock"""
    # pass


# def plot_scatter_matrix():
    # """Creating a scatter matrix of stocks with histograms for distributions"""
    # pass 


# def plot_heat_map():
    # """Create a heat map for the correlation matrix"""
    # pass 


# def plot_efficient_frontier():
    # """Plot efficient frontier with the highest Sharpe and Sortino Ratio"""
    # pass 


# def create_dashboard():
    # """Pack all the visualizations within a dashboard """
    # pass


def process(ticker_symbols_list, weights):
    main_df, ticker_symbols_list = extract_data(ticker_symbols_list)

    weights = [float(weight)/100 for weight in weights if weight != ""]
    print(weights)
    for i in range(len(ticker_symbols_list) - len(weights)):
        weights.append(0.0)

    # final_df = clean_nulls(main_df)
    final_df = main_df
    print(final_df.head())
    mean, std = descriptive_stats(final_df)
    
    portfolio_cumulative_returns, ending_cumulative_returns = monte_carlo_simulation(final_df, ticker_symbols_list, mean, std, weights)
    print(portfolio_cumulative_returns.head())

    plot_data_table(final_df)
    plot_returns(final_df)
    plot_monte_carlo(portfolio_cumulative_returns)
    plot_freq_dist_of_last_day(ending_cumulative_returns)
    prob_dist_at_certain_confidence_interval(ending_cumulative_returns)
    plot_exponential_weighted_average(final_df)
    plot_rolling_standard_deviation(final_df)
    plot_to_show_risk(final_df)
    plot_daily_returns(final_df)

    print('Finished Running Optimizations')


if __name__ == '__main__':
    app = user_interface.UI()
    app.mainloop()