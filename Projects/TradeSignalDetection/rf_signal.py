import pandas as pd 
import sklearn as sk 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from pandas_datareader import data
from trader import RF
from metrics import sharpe_ratio, drawdown

def fetch_data(ticker):
    if type(ticker) is not str:
        raise ValueError("'ticker' variable needs to be in string format.")
    
    return data.DataReader(ticker, 'yahoo', start_date, end_date)


start_date = '2012-01-01'
end_date = '2020-02-29'

sbi = fetch_data('SBIN.NS')
infy = fetch_data('INFY.NS')
reliance = fetch_data('RELIANCE.NS')
cipla = fetch_data('CIPLA.NS')
tcs = fetch_data('TCS.NS') 
sunpharma = fetch_data('SUNPHARMA.NS')

stocks = [sbi, infy, reliance, cipla, tcs, sunpharma]
tickers = ['SBIN.NS','INFY.NS','RELIANCE.NS','CIPLA.NS','TCS.NS', 'SUNPHARMA.NS']
for stock, ticker in zip(stocks, tickers):
    roc = stock['Close'].pct_change()

    trade = RF(stock)
    training_data = trade.generate_data(roc, 0.04)
    predicted_data = trade.train_model(training_data)
    equity, shares, gains = trade.generate_trade_strategy(predicted_data, share_count=100, balance=100000)
    sharpe_ratio(equity)
    drawdown(equity)

    #Equity plot
    plt.plot(equity)
    plt.title("Gain in equity of {}".format(ticker))
    plt.ylabel("Change in Equity")
    plt.show()

    #Gain plot 
    plt.plot(gains)
    plt.title("Gain rate of {}".format(ticker))
    plt.ylabel("Gain value")
    plt.show()