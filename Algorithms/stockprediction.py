import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import pandas_datareader as web
import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators

start = datetime.datetime(2008, 1, 1)
end = datetime.datetime(2018, 12, 8)

df = pd.read_csv('/home/sauraj/spy.csv')

#Exponential Moving Average feature 
df['5-DayEMA'] = df['Adj Close'].ewm(span=5).mean()
df['20-DayEMA'] = df['Adj Close'].ewm(span=20).mean()

df_dummy['CCI-pct%'] = df_dummy['CCI'].pct_change() * 100
#Pseudocode for trend labelling 
#5 Day EMA should be greater than 20 Day EMA to signal an uptrend, and vice versa (Will change these to 9/14 if needed)
#MACD value should be greater than the Signal line to signal an uptrend, and vice versa
#RSI shows a bullish trend between 40-90, hence an uptrend parameter is to be defined for RSI above 40
#RSI shows a bearish trend between 10-60, hence a downtrend parameter is to be defined for RSI below 40
#CCI has to be implemented using a CCI-pct% feature column
for i in range(n, k):
    if(df_dummy['5-DayEMA'][i] > df_dummy['20-DayEMA'][i]) & (df_dummy['MACD'][i] > df_dummy['MACD_Signal'][i]) &
    (df_dummy['RSI'][i] > 40):

        df_dummy['Trend'][i] = 'Bullish'

    elif(df_dummy['5-DayEMA'][i] < df_dummy['20-DayEMA'][i]) & (df_dummy['MACD'][i] < df_dummy['MACD_Signal'][i]) &
    (df_dummy['RSI'][i] < 40):

        df_dummy['Trend'][i] = 'Bearish'
    #Would like some help on using CCI as a parameter for labelling because for now I am only using pct.change but it 
    #might not be feasible in all rows.
    elif(df_dummy['CCI-pct%'][i] < -100):

        df_dummy['Trend'][i] = 'Reversal'
       
    else:
        df_dummy['Trend'][i] = 'Reversal'
    #elif(df_dummy['5-DayEMA'][i] < df_dummy['20-DayEMA'][i]) | (df_dummy['MACD'][i] > df_dummy['MACD_Signal'][i]):
        #df_dummy['Trend'] = 'Reversal'
    
    
"""
This snippet I wrote here has been fetched from a research paper I read for literature review and cites that we can label 
bullish, bearish or reversal with Stochastic Oscillator according to these rules
"""

#Pseudocode of Stochastic Oscillator I read in the research paper 
if %K(t-1) < %D(t-1) < 20 and %D(t) < %K(t) < 20:
    df['Trend'][i] = 'Bullish'
    
elif %K(t-1) > %D(t-1) > 80 and %D(t) > %K(t) > 80:
    df['Trend'][i] = 'Bearish'

else:
    df['Trend'][i] = 'Reversal'
    
"""
The only problem which I face here is hardcoding this rule based algo with the algo I designed above. Would like your help as to
how I can use this for better feature labelling as "Bullish", "Bearish" or "Reversal"
