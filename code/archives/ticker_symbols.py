
from pandas_datareader import data as pdr
import datetime

def extract_data(tickers, start_date, end_date):
    df = pdr.get_data_yahoo(tickers, start_date, end_date)
    return df  

# df = extract_data(['GOOG', 'AAPL'], datetime.datetime(2019, 9, 10), datetime.datetime(2019, 10, 1))
