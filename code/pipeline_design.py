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

import ticker_symbols


##############
# FRONT END UI
##############
"""Creation of front end frame with various object components"""
class UI(tk.Tk):
    """
    Main Class that initiates and control the UI
    """
    def __init__(self, *args, **kwargs) :
        tk.Tk.__init__(self, *args, **kwargs)

        # self.attributes('-fullscreen', True)

        # Create a container frame to hold all the pages inside it
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = OrderedDict()
        self.current_frame = None

        self.create_frames(container)

        self.frames_list = list(self.frames.keys())
        self.current_frame_index = 0
        self.show_frame('StartPage')

    def create_frames(self, container):
        start_page_frame = StartPage(container, self)
        self.frames['StartPage'] = start_page_frame
        start_page_frame.grid(row=0, column=0, sticky="nsew")

    def show_frame(self, cont, data=None):
        """
        Display the given frame
        :param cont:
        :return:
        """
        frame = self.frames[cont]
        self.current_frame = cont
        print("current frame is", cont)
        frame.tkraise()
        return frame

    def reset(self):
        self.show_frame(self.frames_list[0])



class StartPage(tk.Frame):
    """
    Start page frame Class
    """
    def __init__(self, parent, controller):
        self.controller = controller
        tk.Frame.__init__(self, parent)

        menubar = MenuBar(self, controller)
        controller.config(menu=menubar)

        head_frame = HeadFrame(self, controller)
        self.body_frame = BodyFrame(self, controller)
        self.bottom_frame = PortfolioAssetsFrame(self, controller)
        head_frame.pack()
        self.body_frame.pack(fill=tk.BOTH, expand=True)
        self.bottom_frame.pack(fill=tk.BOTH, expand=True, padx=15)

        optimize_btn = tk.Button(self, text="Optimize", bg='blue', fg='white', command=self.optimize)
        optimize_btn.pack(pady=10)

    def optimize(self):
        print("Getting Values")
        data = {}
        data['time'] = time.strftime("%H-%M")
        data['investor_profile'] = {'high_risk_tolerance':self.body_frame.checkcmd1.get(),
                                    'low_risk_tolerance':self.body_frame.checkcmd2.get()}
        data['asset_type'] = self.body_frame.radiocmd.get()
        data['investment_amount'] = self.body_frame.scale1.get()
        data['retirement_time_horizon'] = self.body_frame.scale2.get()
        data['tickers'] = [[e.get() for e in row] for row in self.bottom_frame.asset_rows]
        print(data)
        ticker_symbols_list = [l[0] for l in data['tickers']]
        print(ticker_symbols_list)
        df = ticker_symbols.extract_data(ticker_symbols_list, datetime.datetime(2019, 9, 10), datetime.datetime(2019, 10, 1))
        print(df)



class HeadFrame(tk.Frame):
    def __init__(self, parent, controller):
        self.controller = controller
        self.parent = parent
        tk.Frame.__init__(self, parent)

        self.title_label = tk.Label(self, text="PORTFOLIO OPTIMIZER", font=('times', 30, 'bold'))

        self.watch_label = tk.Label(self, font=('times', 12))
        self.update_watch()

        logo = Image.open("logo.jpg")
        self.logo = ImageTk.PhotoImage(logo)
        self.logo_label = tk.Label(self, image=self.logo)

        self.columnconfigure(0, weight=4)
        self.columnconfigure(1, weight=1)

        self.title_label.grid(row=0, column=0, sticky='s')
        self.watch_label.grid(row=1, column=0, sticky='n')
        self.logo_label.grid(row=0, rowspan=2, column=1, sticky='n')


    def update_watch(self): 
        time_string = "Date: " + time.strftime('%d-%m-%Y') + "  Time: " + time.strftime('%H:%M:%S')
        self.watch_label.configure(text=time_string)
        self.controller.after(200, self.update_watch) #it'll call itself continuously


class BodyFrame(tk.Frame):
    def __init__(self, parent, controller):
        self.controller = controller
        self.parent = parent
        tk.Frame.__init__(self, parent)

        label_1 = tk.Label(self, text="Investor Profile", borderwidth=2, relief="solid", height=3)
        label_2 = tk.Label(self, text="Asset Type", borderwidth=2, relief="solid", height=3)
        label_3 = tk.Label(self, text="Investment Amount", borderwidth=2, relief="solid", height=3)
        label_4 = tk.Label(self, text="Retirement Time Horizon", borderwidth=2, relief="solid", height=3)
        label_1.grid(row=0, column=0, rowspan=2, sticky='nswe', pady=10, padx=(15, 75))
        label_2.grid(row=2, column=0, rowspan=2, sticky='nswe', pady=10, padx=(15, 75))
        label_3.grid(row=4, column=0, rowspan=2, sticky='nswe', pady=10, padx=(15, 75))
        label_4.grid(row=6, column=0, rowspan=2, sticky='nswe', pady=10, padx=(15, 75))

        self.checkcmd1 = tk.IntVar()
        self.checkcmd2 = tk.IntVar()
        self.checkcmd1.set(0)
        self.checkcmd2.set(0)
        checkbox1 = tk.Checkbutton(self, variable=self.checkcmd1, onvalue=1, offvalue=0, text="High Risk Tolerance")
        checkbox2 = tk.Checkbutton(self, variable=self.checkcmd2, onvalue=1, offvalue=0, text="Low Risk Tolerance")
        checkbox1.grid(row=0, column=1, columnspan=2, sticky='ws')
        checkbox2.grid(row=1, column=1, columnspan=2, sticky='wn')

        self.radiocmd = tk.IntVar()
        radio1 = tk.Radiobutton(self, text="Stocks", variable=self.radiocmd, value=1)
        radio2 = tk.Radiobutton(self, text="Bonds", variable=self.radiocmd, value=2)
        radio3 = tk.Radiobutton(self, text="Equities", variable=self.radiocmd, value=3)
        radio4 = tk.Radiobutton(self, text="Fixed Income", variable=self.radiocmd, value=4)
        radio1.grid(row=2, column=1, sticky='sw')
        radio2.grid(row=2, column=2, sticky='sw')
        radio3.grid(row=2, column=3, sticky='sw')
        radio4.grid(row=2, column=4, sticky='sw')

        self.scale1_label = tk.Label(self, text="$0",relief="solid", height=2)
        self.scale1 = tk.Scale(self, orient='horizontal', from_=0, to=1000000, command=self.set_scale1)
        self.scale1_label.grid(row=4, column=1, sticky='sew')
        self.scale1.grid(row=4, column=2, columnspan=2, sticky='se')

        self.scale2_label = tk.Label(self, text="0",relief="solid", height=2)
        self.scale2 = tk.Scale(self, orient='horizontal', from_=0, to=50, command=self.set_scale2)
        self.scale2_label.grid(row=6, column=1, sticky='sew')
        self.scale2.grid(row=6, column=2, columnspan=2, sticky='se')

    def set_scale1(self, event):
        self.scale1_label.configure(text="$"+event)
        self.controller.update()
    
    def set_scale2(self, event):
        self.scale2_label.configure(text=event)
        self.controller.update()


class PortfolioAssetsFrame(tk.Frame):
    def __init__(self, parent, controller):
        self.controller = controller
        self.parent = parent
        tk.Frame.__init__(self, parent, highlightbackground="black", highlightthickness=2)

        titles = ["Number", "Ticker Symbol", "Allocation/Weight (%)", "Min-weight (%)", "Max-weight (%)"]
        for i in range(len(titles)):
            self.columnconfigure(i, weight=1)

        for i in range(len(titles)):
            label = tk.Label(self, text=titles[i], font=('times', 16), borderwidth=2, height=3)
            label.grid(row=0, column=i, padx=(15, 0), sticky='news')

        nrows = 5
        self.asset_rows = []
        for i in range(nrows):
            self.add_asset_row(i+1)

        
    def add_asset_row(self, pos):
        number_label = tk.Label(self, text=str(pos), font=('times', 16), borderwidth=2, height=1)
        number_label.grid(row=pos, column=0, padx=15, sticky='nsew')

        entry_fields = []
        n_entries = 4 
        for i in range(n_entries):
            entry = tk.Entry(self, width=15, highlightbackground='black', highlightthickness=1)
            entry.grid(row=pos, column=i+1)
            entry_fields.append(entry) 
        self.asset_rows.append(entry_fields)     

        
class MenuBar(tk.Menu):
    def __init__(self,master,controller, **kw):
        tk.Menu.__init__(self, master, kw)

        filemenu = FileMenu(self, controller)
        self.add_cascade(label="File", menu=filemenu)

        windowsmenu = WindowsMenu(self, controller)
        self.add_cascade(label="Windows", menu=windowsmenu)

        version = VersionMenu(self, controller)
        self.add_cascade(label="Version", menu=version)


class FileMenu(tk.Menu):
    def __init__(self, master, controller, **kw):
        tk.Menu.__init__(self, master, kw, tearoff=0)

        self.add_command(label="New", command=self.new)
        self.add_command(label="Open", command=self.open)
        self.add_separator()
        self.add_command(label="Exit", command=controller.quit)

    def new(self):
        print("new")

    def open(self):
        print("open")

class WindowsMenu(tk.Menu):
    def __init__(self, master, controller, **kw):
        tk.Menu.__init__(self, master, kw, tearoff=0)

        self.add_command(label="window 1", command=self.window_one)
        self.add_separator()
        self.add_command(label="window 2", command=self.window_two)

    def window_one(self):
        print("window one")

    def window_two(self):
        print("window two")

class VersionMenu(tk.Menu):
    def __init__(self, master, controller, **kw):
        tk.Menu.__init__(self, master, kw, tearoff=0)

        self.add_command(label="Help Index", command=self.help_index)
        self.add_command(label="About...", command=self.about)

    def help_index(self):
        print("Help Index")

    def about(self):
        print("about")


app = UI()
app.mainloop()

#################
# DATA INGESTION
#################
class Ingestion(self):
    """Extracts data from an external source into a python dataframe"""

    def extract_data():
        """Creates an api connection with yahoo finance via pandas datareader"""
        pass  


###################
# DATA MANIPULATION 
###################
class Manipulation(self):
    def clean_nulls():
        """Remove nulls from the dataframe"""
        pass

    def rename_columns():
        pass

    def missing_week_check():
        """Check for missing weeks of data and adjust"""
        pass
    
    def apply_week_normalization():
        """For stocks that do not have the max week impute with mean"""
        pass
    
    def apply_pivot_function():
        """Pivot the dataframe for analysis"""
        pass


##########################
# LOADING HISTORICAL DATA
#########################
class LoadingDatabase(self):
    def create_connection_string():
        pass 
    
    def create_cursor():
        pass 
    
    def upload_data():
        pass 


########################
# DESCRIPTIVE STATISTICS
########################
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
        """Calculate rolling betas 7-30-180"""
        pass


###################################################
# OPTIMAL PORTFOLIO SELECTION AND ASSET FORECASTING
##################################################
class Analysis(self):
# Optimal Portfolio
    def portfolio_optimization_risk_return():
        """Drop correlated stocks and ones that have high volatility"""
        pass 
        
    def daily_returns():
        """Calculate daily returns of all stocks"""
        pass 

    def portfolio_returns():
        """Add weights to each stock within the portfolio and calculate returns"""
        pass

    def monte_carlo_weights():
        """Run portfolio returns with random weights"""
        pass

    def efficient_frontier():
        """Create efficient frontier based on monte carlo simulation to pick best portfolio mix"""
        pass

    def calculate_confidence_intervals():
        """Calculate confidence intervals for return"""
      
# Forecasting returns for each stock within the portfolio
    def calc_portfolio_returns_compared_to_benchmark():
        """For a specific investment calculate cumulative portfolio returns to the sp_500"""
        pass 
        
    def monte_carlo_returns():
        """Run monte-carlo simulation of each stock return"""
        pass


#########################
# DASHBOARD VISUALIZATION
#########################
class Visualization(self):
    """Create visualizations of all analysis and pack into a dashboard"""

    def plot_prediction():
        """Plotting prediction for 6 months to a year for each stock"""
        pass

    def plot_scatter_matrix():
        """Creating a scatter matrix of stocks with histograms for distributions"""
        pass 

    def plot_heat_map():
        """Create a heat map for the correlation matrix"""
        pass 

    def plot_efficient_frontier():
        """Plot efficient frontier with the highest Sharpe and Sortino Ratio"""
        pass 

    def create_dashboard():
        """Pack all the visualizations within a dashboard """
        pass

##################################
# CONNECTING OUTPUT TO A FRONT-END 
##################################




###############################
# ENTIRE PROCESS LOGIC SEQUENCE
###############################
def main():
    """Logic for entire process"""
    pass





if __name__ == '__main__':
    main() 

