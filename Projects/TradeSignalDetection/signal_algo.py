import pandas as pd 
import sklearn as sk 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from pandas_datareader import data
from trader import PCARF, RF, PCABag, SVCModel
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
tcs = fetch_data('TCS.NS') 

stocks = [sbi, infy, reliance, tcs]
tickers = ['SBIN.NS','INFY.NS','RELIANCE.NS','TCS.NS']
equity_change, sharpe, draw_down = {}, {}, {}
for stock, ticker in zip(stocks, tickers):
    roc = stock['Close'].pct_change()

    trade = PCARF(stock)
    training_data = trade.generate_data(roc, 0.03)
    predicted_data = trade.train_model(training_data)
    equity, shares, gains = trade.generate_trade_strategy(predicted_data, share_count=100, balance=100000)
    ratio_sharpe = sharpe_ratio(equity)
    dd = drawdown(equity)

    #Store values in tables 
    equity_change[ticker] = equity
    sharpe[ticker] = ratio_sharpe
    draw_down[ticker] = dd


pd.DataFrame(equity_change).plot()
plt.tight_layout()
plt.show()
print(sharpe)
print(draw_down)