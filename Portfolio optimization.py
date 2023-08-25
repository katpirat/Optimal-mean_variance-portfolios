
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import datetime as dt
import yfinance as yf
from numpy.linalg import inv

class mean_variance_portfolio: 
    def __init__(self, ticker_list, n_pfs = 1000, start_date = '2010-10-20', end_date = '2020-10-20') -> None:
        self.ticker_list = ticker_list
        dat = yf.download(ticker_list, start = start_date, end = end_date)
        self.stock_data = dat['Close']
        self.returns = self.stock_data.pct_change()
        self.mus = (1 + self.returns.mean())**252 -1
        self.cov = self.returns.cov()*252
        self.n_pfs = n_pfs

    
    def m_var(self, n_assets = 5): 
        mean_var_pairs = []
        np.random.seed(75)
        assets = np.random.choice(list(self.returns.columns), n_assets, replace=False)

        for i in range(self.n_pfs):
            assets = np.random.choice(list(self.returns.columns), n_assets, replace=False)
            weights = np.random.rand(n_assets)
            weights = weights/sum(weights)
            portfolio_E_Variance = 0
            portfolio_E_Return = 0
            for i in range(len(assets)):
                portfolio_E_Return += weights[i] * self.mus.loc[assets[i]]
                for j in range(len(assets)):
                    portfolio_E_Variance += weights[i] * weights[j] * self.cov.loc[assets[i], assets[j]]

            mean_var_pairs.append([portfolio_E_Return, portfolio_E_Variance])

        mean_variance_pairs = np.array(mean_var_pairs)

        sigma = self.cov
        ones = np.ones(len(self.ticker_list))

        A = np.array([
            [self.mus.T @ inv(sigma) @ self.mus, self.mus.T @ inv(sigma) @ ones],
            [self.mus.T @ inv(sigma) @ ones, ones.T @ inv(sigma) @ ones]
        ])

        m_list = np.linspace(0.0, 0.6)
        s = self.sigma_min(m_list, A = A )

        m,v = zip(*mean_variance_pairs)

        plt.scatter(v, m)
        plt.plot(s, m_list, 'r-')
        plt.show()

    def sigma_min(self, mu, A):
            return (A[0,0] - mu * 2 * A[0,1] + A[1,1] * mu **2)/(A[0,0] * A[1,1] - A[0, 1]**2)

### Example of usage ### 
### Create an object with a list of ticker symbols as argumemnt. Here we will use the largest companies in the SP500 index. 

sp500_top_20_tickers = [
    "AAPL",
    "MSFT",
    "AMZN",
    "TSLA",
    "GOOGL",
    "GOOG",
    "JNJ",
    "JPM",
    "V",
    "PG",
    "MA",
    "UNH",
    "BAC",
    "WMT",
    "INTC",
    "HD",
    "NVDA",
    "VZ"
]

start, end = '2010-10-20', '2020-10-20'

n_perfs = 2000 # Amount of permutations done when randomly sampling the stocks to create portfolios. 

obj1 = mean_variance_portfolio(sp500_top_20_tickers, n_pfs= n_perfs, start_date=start, end_date=end)
obj1.m_var(10) # Argument: Number of different stocks in the random permutation procedure to create portfolios. 

