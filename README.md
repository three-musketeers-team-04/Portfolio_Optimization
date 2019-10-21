![Company logo](code/logo.jpg)

# Portfolio Optimizer (PO)

## Portfolio optimizer suggest an ideal allocation of stocks within your portfolio
* PO provides a portfolio analysis based on investor profile, investment amount, investment time horizon and stock selection   
* Calculations includes 20 years of historical stock data
* One can input any different combinations of stocks and run the optimizer multiple times to get different reports
* Below is a screenshot of the user interface:

![User interface](code/images/user_interface.png)


## How it works
1. Clone the repo to your local drive
2. Locate the code folder within the repo and within your terminal type python main.py <Hit Run>
3. Once you hit Run the PO pops up. Select Investor Profile, asset type (stocks), investment amount, investment time horizon, and enter stock and allocation details
4. Hit Optimize once all the details are entered and the PO will run all the anlalysis on the backed to generate a report in the report folder
5. Each report is time_stamped so generate as many reports you want with different portfolios


## Usage
The optimimzer can be used by anyone who wants to find out what is the best allocation for their stocks 
The portfolio analysis report has the below sections:
* **Returns**: Analyzes historical data to provide confidence interval for current investment over the investment horizon
* **Comparisons**: Compares your current portfolio with the S&P500 based on historical returns
* **Optimized portfolio**: Suggests an optimal allocation for your portfolio based on your slected risk level  


## Requirements and Configuration
* All requirements and dependencies are in code/requirements folder 
* The only other configuration is to run the wkhtmltox-0.12.5-1.msvc2015-win64.exe file in the code/archives folder
* Once the config is run the user needs to add the path to their System - path variable. Below is an image of where one could find that:

![Environment variable config](code/images/add_to_path_variable.png)

Have fun with the portfolio optimizer! :+1:

## Disclaimer
Developers of the PO are not registered as a securities broker-dealer or an investment adviser either with the U.S. Securities and Exchange Commission (the “SEC”) or with any state securities regulatory authority. We are neither licensed nor qualified to provide investment advice. Use PO suggestions at your own discretion