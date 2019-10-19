# Analysis outline on final_df
## :param: df, weights, max, min, investment, profile, asset_type 
## :return: optimized portfolio, ratios, prediction trend, investment          range confidence

## Basic stats
* Calculate daily_returns_df
* Calculate mean_return_series
* Calculate std_return_series
* Risk_free_rate

## Beta calculations
* Calculate portfolio cov matrix
* Calculate portfolio var
* Calculate portfolio beta

## Monte Carlo Simulation
### Investment range for current portfolio
* :param: original_weights, simulated_returns, time_horizon, investment
* Calculate simulated_returns 
* Calculate simulated_portfolio_daily_returns
* Calculate simulated_cumulative_portfolio_returns
* Calculate ending_cumulative_portfolio_returns
* Calculate probability distribution of ending_cumulative_portfolio_returns
* Calculate confidence interval of ending_cumulative_porfolio_return
* Apply interval to a investment amount 

### Investment range for future portfolio with weight changes [Efficient frontier]
* :param: simulated_weights, simulated_returns, time_horizon, investment, risk-free investor
* Calculate random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)
* Display simulated_random(num_portfolios, mean_returns, cov_matrix, risk_free_rate) [Maximum Sharpe (high-risk) and Minimum Volatility (low-risk)]
* Are we applying any weights? If then rational for high-risk vs low-risk check

## Calculate ratios and returns
* Sharpe ratio(mean_daily_returns, covariance, risk_free_rate)
* Sortinos ratio()
* Coeff of variatoin()
* Returns
* CAGR
* Portfolio_std_deviation
* Volatility
* Correlation_coefficient

## Plots
* portfolio_cumulative_returns
* ending_cumulative_returns
* prob_dist_ending_cumulative_returns
* exponential_weighted_average
* portfolio_beta
* rolling_standard_deviation
* efficient frontier
* correlation matrix 
* 6 months-1 year forecasting trend
* table of ratios