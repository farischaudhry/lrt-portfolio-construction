import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

from Portfolio import Portfolio
from EqualWeightedPortfolios import *
from HRPPortfolios import *
from ClusteringPortfolios import *
from MomentumPortfolios import *
from MeanReversionPortfolios import *
from MarkowitzPortfolios import *
from BlackLittermanPortfolios import *

def get_data(isOneValid=False):
    assets = ['EETH-USD', 'RSETH-USD', 'UNIETH-USD', 'PUFETH-USD', 'EZETH-USD', 'RSWETH-USD', 'WEETH-USD']
    end_date = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(assets, start='2020-01-01', end=end_date)['Adj Close']

    # If isOneValid is True, only one asset must be valid for the row to be kept.
    if isOneValid:
        data = data.dropna(axis=1, how='all')
    else:
        data = data.dropna()
    return data

def main():
    data = get_data()

    portfolios = [
        EqualWeightedPortfolio(data),
        # HRPPortfolio(data)
        # HRPLongShortPortfolio(data),
        # EqualWeightLongShortPortfolio(data),
        # MomentumLongShortPortfolio(data),
        # MeanReversionLongShortPortfolio(data),
        # KMeansPortfolio(data),
        # DBSCANPortfolio(data),
        # MinimumVariancePortfolio(data),
    ]

    for portfolio in portfolios:
        cumulative_returns = portfolio.try_strategy()
        plt.plot(cumulative_returns, label=portfolio.__class__.__name__)

    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
