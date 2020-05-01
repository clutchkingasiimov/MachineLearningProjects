import pandas as pd 
import numpy as np 


def sharpe_ratio(returns):
    """
    Calculates the Sharpe Ratio of the equity
    """
    returns = pd.Series(returns)
    pct_returns = returns.pct_change()
    sharpe = np.sqrt(len(returns)) * (pct_returns.mean() / pct_returns.std())
    return sharpe


def drawdown(returns):
    
    peak, trough = max(returns), min(returns)
    dd = (peak-trough)/peak * 100 
    return dd
    