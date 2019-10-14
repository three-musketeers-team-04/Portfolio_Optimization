###########################
# STOCK PORTFOLIO OPTIMIZER
###########################
# Customer input: Inputs upto 10 stock tickers of a certain industry e.g. ['Tech', 'Advertising', 'Finance]

# PO performs the below tasks:
# Ingests historical data upto 5 years
# Processes and manipulates data
# Stores it into a database
# Calculates different statistics, returns, and risk
# Selects stocks to keep in portfolio based on various variables? - Volatility, Scoring of each stock, and monte carlo simulation - confidence interval (return)
# Visualizes an optimized porfolio, return, and dividends
# Built in trailing stop for certain thresholds
# How do we run the monte carlo simulation on scores instead of prices?

class Ingestion(self):
    """Extracts data from an external source into a python dataframe"""

    def create_api():
    """Creates an api key with yahoo finance"""
        pass  


class Processing(): 
    def clean_nulls():
    """Remove nulls from the dataframe"""
        pass 

    def remove_outliers():
        pass 
    
    def remove_duplicates():
        pass


class Manipulation(self):
    def rename_columns():
        pass

    def apply_function():
        pass


class LoadingDatabase(self):
    def create_connection_string():
        pass 
    
    def create_cursor():
        pass 
    
    def upload_data():
        pass 


class Reporting(self):
    def descriptive_stats():
    """Descriptive stats correlation, mean, standard deviaiton"""    
        pass
    
    def rolling_statistics():
    """ Calculate rolling mean, standard deviation at different levesl 7-30-180 """
        pass

    def beta():
    """Calculate covariance, variances, and beta values"""
        pass   

    def rolling_betas():
    """Calculate rolling beats 7-30-180"""
        pass


class Analysis(self):
    def daily_returns():
    """Calculate daily returns of all stocks"""
        pass 

    def portfolio_returns():
    """Add weights to each stock within the portfolio and calculate returns"""
        pass

    def cum_portfolio_returns_compared_to_sp():
    """For a specific investment calculate cumulative portfolio returns to the sp_500"""
        pass 
    
    def portfolio_optimization_risk_return():
    """Drop correlated stocks and ones that have high volatility""""
        pass 
    
    def calculate_optimized_return():
    """Re-calculate based on optimized return"""
    
    def monte_carlo():
    """Run monte-carlo simulation"""
        pass

    def linear_regression():
        pass

    def knn():
        pass 

    def support_vector_machines():
        pass

    def test_models():
        pass

    def cross_validate():
        pass 

    def predict():
        pass


class Visualization(self):
    def plotting_prediction():
        pass

    def create_scatter_matrix():
        pass 

    def create_heat_map():
        pass 

    def create_dashboard():
        """Creates a dashboard of all components"""
        pass


if __name__ == '__main__':
    pass 

