![Company logo](code/logo.jpg)

# Portfolio Optimizer (PO)
![Python](https://camo.githubusercontent.com/de59e8e9b410aa0b9479b114040c06468ef33cfc/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d76332e362b2d626c75652e737667)  ![Build Status](https://travis-ci.org/anfederico/Clairvoyant.svg?branch=master)  ![Dependecies](https://camo.githubusercontent.com/6266857d1c53194119edf1d9aafae7a4b301fa16/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f646570656e64656e636965732d7570253230746f253230646174652d627269676874677265656e2e737667) ![Contributions Welcome](https://camo.githubusercontent.com/72f84692f9f89555c176bb9e0eca9cf08d97fec9/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636f6e747269627574696f6e732d77656c636f6d652d6f72616e67652e737667) ![MIT license](images/mit-license.svg)

## Portfolio optimizer suggest an ideal allocation of stocks within your portfolio
* PO provides a portfolio analysis based on investor profile, investment amount, investment time horizon and stock selection
* Calculations includes 20 years of historical stock data
* One can input any different combinations of stocks and run the optimizer multiple times to get different reports within the reports folder
* Below is a screenshot of the user interface:

![User interface](code/images/user_interface.png)


## How it works
1. Clone the repo to your local drive
2. From your terminal locate the code folder within the repo and within your terminal type '''python main.py''' <Hit Run>
3. Once you hit Run the PO pops up. Select Investor Profile, asset type (stocks), investment amount, investment time horizon, and enter stock and allocation details
4. Hit Optimize once all the details are entered and the PO will run all the anlalysis on the backed to generate a report in the report folder
5. Each report is time_stamped so generate as many reports you want with different portfolios


## Usage
The optimimzer can be used by analysts who want to do a quick discovery on basic stats and graphs as well as investors who want to understand what is the best allocation of their stock portfolio based on customized inputs
The portfolio analysis report has the below sections:
* **Portfolio Overview**: Analyzes historical data to provide confidence interval for returns based on current investment amount over the investment horizon. Besides that one can check for basic graphs (e.g. Box plots, histograms, correlation matrices)
* **Portfolio Analysis**: Suggests an optimal allocation for your portfolio based on your selected risk level. In addition the portfolio optimizer provides annualized returns and volatility on individual stocks within the portfolio    


## Requirements and Configuration
* All requirements and dependencies are in the code/requirements folder 
* The only other configuration is to run the wkhtmltox-0.12.5-1.msvc2015-win64.exe file in the code/archives folder
* Once the config is run the user needs to add the path to their System - path variable. Below is an image of where one could find that:

![Environment variable config](code/images/add_to_path_variable.png)

Have fun with the portfolio optimizer! :+1:


## Disclaimer
Developers of the PO are not registered as a securities broker-dealer or an investment adviser either with the U.S. Securities and Exchange Commission (the “SEC”) or with any state securities regulatory authority. We are neither licensed nor qualified to provide investment advice. Use PO suggestions at your own discretion