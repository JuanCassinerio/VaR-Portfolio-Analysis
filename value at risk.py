"""
Value at Risk
Higher expected loss given a Degree of probability %, taking into account the weighted portafolio assets distribution
"""

import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.cm as cm


#portfolio and data to use for analysis
portfolio_value = 1
tickers = ['SPY', 'BND', 'GLD', 'QQQ', 'VTI']
weights = np.array([1/len(tickers)] * len(tickers))

years = 15
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365*years)

adj_close_df = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start=startDate, end=endDate)
    adj_close_df[ticker] = data['Adj Close']

#portfolio past behaviour

log_returns = np.log(adj_close_df / adj_close_df.shift(1))
log_returns = log_returns.dropna()
historical_returns = (log_returns * weights).sum(axis=1) #portfolio historical log returns

cov_matrix = log_returns.cov() * 252 
portfolio_std_dev = np.sqrt(weights.T @ cov_matrix @ weights) #volatility

'''
given a distribution of returns(with a probability), we get the maximum loss over a probability 
(for 100% accuracy a total loss is always possible(given the normal distribution back arm))
'''

confidence_levels = [0.90, 0.95, 0.99]
days = 5

VaRs = []
for cl in confidence_levels:
    VaR = portfolio_value * (norm.ppf(1 - cl) * portfolio_std_dev * np.sqrt(days / 252) - historical_returns.mean() * days)
    VaRs.append(VaR)

print("Confidence Level / Value at Risk")

for cl, VaR in zip(confidence_levels, VaRs):
    print(cl ,"/", round(VaR*100,2),"%")

historical_x_day_returns = historical_returns.rolling(window=days).sum() #log return properties, start to end log return
historical_x_day_returns_dollar = historical_x_day_returns * portfolio_value


plt.hist(historical_x_day_returns_dollar, bins=50, density=True, alpha=0.5, label=f'{days}-Day Returns')
color_map = cm.get_cmap('plasma')  
for i, (cl, VaR) in enumerate(zip(confidence_levels, VaRs)):
  color = color_map(i / len(confidence_levels))  
  plt.axvline(x=VaR, linestyle='--', color=color, label='VaR at {}% Confidence'.format(int(cl * 100)))

plt.xlabel(f'{days}-Day Portfolio Return')
plt.ylabel('Frequency')
plt.title(f'Distribution of Portfolio {days}-Day Returns and Parametric VaR Estimates')
plt.legend()
plt.show()






