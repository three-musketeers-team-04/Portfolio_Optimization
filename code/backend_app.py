#################
# LIBRARY IMPORTS 
#################
import pandas as pd
import numpy as np
import time
import datetime
import pandas_datareader as pdr
import subprocess
from collections import OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored
import scipy.optimize as sco
from jinja2 import Environment, FileSystemLoader
import pdfkit
# from xhtml2pdf import pisa

plt.style.use('fivethirtyeight')
np.random.seed(777)

def extract_data(ticker_symbols_list):
    """Creates an api connection with yahoo finance via pandas datareader"""
    start_sp = datetime.datetime(1999, 10, 1)
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
    cov_matrix = returns_df.cov()
    # Loop through the columns to get mean returns and std dev for each stock variable
    # stock_mean_returns = mean_returns['stock']
    return returns_df, mean_returns_series, std_dev_series, cov_matrix 
    

# Forecasting returns for each stock within the portfolio
def monte_carlo_simulation(final_df, ticker_symbols_list, avg_daily_returns, std_dev_daily_returns, weights, num_trading_days=3):
    # Set number of simulations and trading days
    num_simulations = 1000
    num_trading_days = 252 * num_trading_days

    # Get last closing prices
    last_prices = final_df.iloc[-1,:]

    avg_daily_returns_list = avg_daily_returns.to_list()
    std_dev_daily_returns_list = std_dev_daily_returns.to_list()

    # Initialize empty DataFrame to hold simulated prices for each simulation
    simulated_price_df = pd.DataFrame()
    simulated_portfolio_cumulative_returns = pd.DataFrame()

    # Run the simulation of projecting stock prices for the next trading year, `1000` times
    for n in range(num_simulations):

        # Initialize the simulated prices list with the last closing prices
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
        simulated_portfolio_daily_returns = simulated_daily_returns.dot(weights)
        
        # Calculate the normalized, cumulative return series
        simulated_portfolio_cumulative_returns[n] = (1 + simulated_portfolio_daily_returns.fillna(0)).cumprod()

    # Print records from the DataFrame
    simulated_portfolio_daily_returns.head()
    simulated_portfolio_cumulative_returns.head()
    simulated_ending_cumulative_returns = simulated_portfolio_cumulative_returns.iloc[-1, :]
    return simulated_portfolio_daily_returns, simulated_portfolio_cumulative_returns, simulated_ending_cumulative_returns


def probability_distribution(ending_cumulative_returns, portfolio_daily_returns):
    ending_cumulative_returns.value_counts(bins=10) / len(ending_cumulative_returns)
    cumulative_returns = (1 + portfolio_daily_returns).cumprod() - 1
    cumulative_returns.head()


def calculate_confidence_intervals(ending_cumulative_returns):
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

####################
# EFFICIENT FRONTIER
####################
def portfolio_annualised_performance(weights, mean_returns_series, cov_matrix):
    port_returns = np.sum(mean_returns_series*weights) *252
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return port_std, port_returns


def random_portfolios(mean_returns_series, cov_matrix, risk_free_rate, num_portfolios):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(mean_returns_series.shape[0])
        weights /= np.sum(weights)
        weights_record.append(weights)
        sim_portfolio_std_dev, sim_portfolio_return = portfolio_annualised_performance(weights, mean_returns_series, cov_matrix)
        results[0,i] = sim_portfolio_std_dev
        results[1,i] = sim_portfolio_return
        results[2,i] = (sim_portfolio_return - risk_free_rate) / sim_portfolio_std_dev
    return results, weights_record

# returns = main_df.pct_change()
# mean_returns = returns.mean()
# cov_matrix = returns.cov()
# num_portfolios = 25000
# risk_free_rate = 0.0178


def display_simulated_ef_with_random(mean_returns_series, cov_matrix, num_portfolios, risk_free_rate, final_df):
    results, weights = random_portfolios(mean_returns_series, cov_matrix, risk_free_rate, num_portfolios)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=final_df.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=final_df.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n") 
    print("Annualised Return:", round(rp,2)) 
    print("Annualised Volatility:", round(sdp,2)) 
    print("\n") 
    print(max_sharpe_allocation) 
    print("-"*80)
    print("Minimum Volatility Portfolio Allocation\n") 
    print("Annualised Return:", round(rp_min,2)) 
    print("Annualised Volatility:", round(sdp_min,2)) 
    print( "\n")
    print(min_vol_allocation) 
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
    plt.savefig('images/efficient_frontier')

###############
def neg_sharpe_ratio(weights, mean_returns_series, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns_series, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns_series, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns_series)
    args = (mean_returns_series, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, mean_returns_series, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns_series, cov_matrix)[0]

def min_variance(mean_returns_series, cov_matrix):
    num_assets = len(mean_returns_series)
    args = (mean_returns_series, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result

def efficient_return(mean_returns_series, cov_matrix, target):
    num_assets = len(mean_returns_series)
    args = (mean_returns_series, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns_series, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients


def display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate, final_df, returns_df):
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=final_df.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    max_sharpe_allocation

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=final_df.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    an_vol = np.std(returns_df) * np.sqrt(252)
    an_rt = mean_returns * 252
    
    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n") 
    print("Annualised Return:", round(rp,2)) 
    print("Annualised Volatility:", round(sdp,2))
    print("\n")
    print(max_sharpe_allocation) 
    print("-"*80)
    print("Minimum Volatility Portfolio Allocation\n") 
    print("Annualised Return:", round(rp_min,2))
    print("Annualised Volatility:", round(sdp_min,2))
    print("\n")
    print(min_vol_allocation)
    print("-"*80)
    print("Individual Stock Returns and Volatility\n")
    for i, txt in enumerate(final_df.columns):
        print(txt,":","annualised return",round(an_rt[i],2),", annualised volatility:",round(an_vol[i],2))
    print("-"*80)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(an_vol,an_rt,marker='o',s=200)

    for i, txt in enumerate(final_df.columns):
        ax.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')
    ax.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    ax.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    target = np.linspace(rp_min, 0.34, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    ax.set_title('Portfolio Optimization with Individual Stocks')
    ax.set_xlabel('annualised volatility')
    ax.set_ylabel('annualised returns')
    ax.legend(labelspacing=0.8)

    data = {}
    an_rt_vol_df = pd.DataFrame()
    for i in range(an_rt.shape[0]):
        an_rt[i] = round(an_rt[i], 2)
        an_vol[i] = round(an_vol[i], 2)
    an_rt_vol_df['Annualised Return'] = an_rt
    an_rt_vol_df['Annualised Volatility'] = an_vol
    data['rp'] = rp
    data['rp_min'] = rp_min
    data['sdp'] = sdp
    data['sdp_min'] = sdp_min
    data['an_rt_vol_df'] = an_rt_vol_df

    return data


# Visualization
def plot_data_table(final_df):
    plt.clf()
    final_df.round(4)*1000
    final_df.head().to_html('images/table.html')
    # table = subprocess.call('wkhtmltoimage -f png --width 0 table.html table.png', shell=True)
    # return table


def plot_individual_stock_trends(final_df):
    plt.clf()
    final_df.plot(subplots = False, figsize = (25,10))
    # final_df.hvplot()
    plt.savefig('images/individual_stock_trends', dpi=100)
    

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


def plot_dist_at_certain_confidence_interval(ending_cumulative_returns):
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


def plot_rolling_standard_deviation(final_df):
    plt.clf()
    final_df.rolling(window=21).std().plot(figsize = (25,10))
    plt.savefig('images/rolling_standard_deviation')


def plot_to_show_risk(final_df):
    plt.clf()
    bplot = final_df.boxplot(figsize = (25,10))
    bplot.axes.set_title("Portfolio Risk", fontsize=12)
    plt.savefig('images/box_plot_for_risk')


def plot_daily_returns(returns_df):
    plt.clf()
    returns_df.plot(subplots = False, figsize = (20,10))
    plt.savefig('images/daily_returns')


def plot_cum_returns(returns_df):
    plt.clf()
    cum_returns_df = (1 + returns_df).cumprod()
    cum_returns_df.plot(subplots = False, figsize = (25,10))
    plt.savefig('images/cum_daily_returns')


def plot_scatter_matrix(final_df):
    """Creating a scatter matrix of stocks with histograms for distributions"""
    plt.clf()
    pd.plotting.scatter_matrix(final_df)
    plt.savefig('images/scatter_matrix')


def plot_scatter_matrix_triangle(final_df):
    plt.clf()
    mask = np.zeros_like(final_df)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(final_df, cmap='RdYlGn', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5)
    plt.yticks(rotation=0) 
    plt.xticks(rotation=90) 
    plt.savefig('images/scatter_matrix_triangle')
    

def plot_heat_map(final_df):
    """Create a heat map for the correlation matrix"""
    fig, ax = plt.subplots(figsize=(14,9))
    plt.title('Heat Map',fontsize=18)
    ax.title.set_position([0.5,1.05])
    ax.set_xticks([])
    sns.heatmap(final_df, annot=True, fmt="", cmap='RdYlGn', ax=ax)
    plt.savefig('images/heat_map')


 # def plot_future_returns_prediction():
    # """Plotting prediction for 6 months to a year for each stock"""
    # pass


# def plot_table_ratios():
#     """Plotting different financial portfolio ratios"""   
#     pass


# def create_dashboard():
    # """Pack all the visualizations within a dashboard """
    # pass


def process(ticker_symbols_list, weights, user_inputs):
    print('\n' * 1)
    print(colored("""
                ########################
                # ASSIGN INPUT VARIABLES
                ########################
              """, 'green'))
    print('\n' * 1)
    main_df, ticker_symbols_list = extract_data(ticker_symbols_list)

    weights = [float(weight)/100 for weight in weights if weight != ""]
    print(weights)
    print('\n' * 1)
    print(ticker_symbols_list)
    for i in range(len(ticker_symbols_list) - len(weights)):
        weights.append(0.0)
    # final_df = clean_nulls(main_df)
    final_df = main_df
    print('\n' * 1)
    
    
    print(colored("""
                ############################
                # INGEST AND MANIPULATE DATA
                ############################
              """, 'green'))
    print('\n' * 1)
    print('Extracting 20 year historical stock closing prices')
    time.sleep(5) # Wait for 5 seconds
    print('\n' * 1)
    print(final_df.head())
    print('\n' * 1)
    
    
    print(colored("""
                ##############
                # ANALYZE DATA
                ##############
              """, 'green'))
    print('\n' * 1)

    returns_df, mean, std, cov_matrix = descriptive_stats(final_df)
    # global final_mean
    final_mean = mean
    
    print('Calculating daily returns')
    print('\n' * 1)
    print(returns_df.head())
    simulated_portfolio_daily_returns, simulated_portfolio_cumulative_returns, simulated_ending_cumulative_returns = monte_carlo_simulation(final_df, ticker_symbols_list, mean, std, weights)
    print('\n' * 1)
    
    print('Calculating simulated portfolio daily returns')
    print(simulated_portfolio_daily_returns.head())
    print('\n' * 1)  
    
    print('Calculating simulated portfolio cumulative returns')
    time.sleep(2) # Wait for 2 seconds
    print('\n' * 1)
    print(simulated_portfolio_cumulative_returns.head())
    print('\n' * 1)

    # returns = main_df.pct_change()
    returns = returns_df
    mean_returns = mean
    # cov_matrix = returns.cov()
    num_portfolios = 25000
    risk_free_rate = 0.0178
    
    print('Calculating simulated effecient frontier with random weights')
    display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, final_df)
    print('\n' * 1)
    print('Calculating simulated effecient frontier with selected weights')
    efficient_frontier_data = display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate, final_df, returns)

 


    
    print(colored("""
                #######################
                # CREATE VISUALIZATIONS
                #######################
              """, 'magenta'))
    print('\n' * 1)
    print('Plotting the data_table')
    plot_data_table(final_df)
    print('\n' * 1)
    print('Potting individual stock trends')
    plot_individual_stock_trends(final_df)
    print('\n' * 1)
    print('Plotting daily returns')
    plot_daily_returns(returns_df)
    print('\n' * 1)
    print('Plotting cumulative returns')
    plot_cum_returns(returns_df)
    print('\n' * 1)
    # print('Plotting exponential weighted average')
    # plot_exponential_weighted_average(final_df)
    # print('\n' * 1)
    # print('Plotting rolling_standard_deviation')
    # plot_rolling_standard_deviation(final_df)
    # print('\n' * 1)
    print('Plotting risk_plot')
    plot_to_show_risk(final_df)


    print('\n' * 1)
    print('Plotting scatter_matrix')
    plot_scatter_matrix(final_df)
    # plot_scatter_matrix_triangle(final_df)
    print('\n' * 1)
    # print('Plotting heat maps')
    # print("\033[1;32;40m Running heat maps \n")
    # plot_heat_map(final_df)
    # print('\n' * 1)


    # print('\n' * 1)    
    print('Plotting simulated portfolio cumulative returns')
    plot_monte_carlo(simulated_portfolio_cumulative_returns)
    print('\n' * 1)
    print('Plotting ending_cumulative_returns')
    plot_freq_dist_of_last_day(simulated_ending_cumulative_returns)
    print('\n' * 1)
    print('Plotting dist_at_confidence_level')
    plot_dist_at_certain_confidence_interval(simulated_ending_cumulative_returns)
    
    
    print(colored("""
                ###################
                # CREATE DASHBOARD
                ###################
              """, 'green'))
    print('\n' * 1)


    # test1 = pn.Column(plot_data_table(final_df), plot_to_show_risk(final_df))
    # test2 = pn.Column(plot_scatter_matrix(final_df), plot_daily_returns(returns_df))
    # dash = pn.Tabs(test1, test2)
    # dash.servable()
    print(colored("""
                ###################
                # OPTIMIZATIONS COMPLETED
                ###################
              """, 'green'))
    print('\n' * 1)

    ## Reporting Section ##
    print("######## Reporting ########")

    # Loading the html template "report.html" to jinja
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template("report.html")

    # Populating a dictionary of values to be passed to the html file
    x = 'x'
    y = 'y'
    cum_returns_pct = '24%'
    template_dict = {"final_df_table":final_df.head().to_html(),
                     "investment_amount":user_inputs['investment_amount'],
                     "investment_horizon":user_inputs['investment_horizon'],
                     'x':x,
                     'y':y,
                     'cum_returns_pct':cum_returns_pct,
                     'rp':efficient_frontier_data['rp'],
                     'rp_min':efficient_frontier_data['rp_min'],
                     'sdp':efficient_frontier_data['sdp'],
                     'sdp_min':efficient_frontier_data['sdp_min'],
                     'an_rt_vol_df':efficient_frontier_data['an_rt_vol_df'].to_html(),


                    }

    # Rendering the template to html passing the dictionary of values to it 
    html_out = template.render(template_dict)

    # Write the html to a file
    out_html_path = 'out.html'
    with open("out.html", "w") as fh:
        fh.write(html_out)

    # Creating a pdf file from the saved output html file
    out_pdf_directory = "reports"
    out_pdf_name = time.strftime('%m-%d-%Y-%H-%M') + "-portfolio-analysis.pdf"
    out_pdf_path = out_pdf_directory + "\\" + out_pdf_name
    table = pdfkit.from_file('out.html', out_pdf_path)

    subprocess.Popen([out_pdf_path], shell=True)