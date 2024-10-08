import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

from Portfolio import Portfolio
from EqualWeightedPortfolios import *
from HRPPortfolios import *
from ClusteringPortfolios import *
from MomentumPortfolios import *
from MeanReversionPortfolios import *
from MarkowitzPortfolios import *
from FactorPortfolios import *

'''
How to use:

- To change the LRTs, change the 'assets' list in the get_data function (must be on Yahoo Finance).
- To change the portfolio strategies, change the 'portfolios' list in the main function.
- To create a new portfolio strategy, create a new class inheriting 'Portfolio' and implement the 'calculate_weights' method, which returns the weights for each asset.
- By default, weights are reblanced every 30 days. To change this, set the 'rebalance_frequency' parameter in the portfolio class constructor.
- Only data up to each rebalance date is used to calculate the weights to prevent look-ahead bias.
- To change between USD to ETH, set inETH=False or True when calling get_data in the main function.
- Bootstraped performance metrics can be enabled by setting bootstrap=True when calling the test_portfolios function.
'''

def get_data(isOneValid=False, inETH=False):
    # ETH-USD is for normalizing the LRTs by ETH-USD price only
    assets = ['ETH-USD', 'EETH-USD', 'RSETH-USD', 'UNIETH-USD', 'PUFETH-USD', 'EZETH-USD', 'RSWETH-USD', 'WEETH-USD']
    end_date = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(assets, start='2020-01-01', end=end_date)['Adj Close']

    # Normalize the LRT prices by ETH-USD price
    if inETH:
        eth_prices = data['ETH-USD']
        data = data.div(eth_prices, axis=0)
    data = data.drop(columns='ETH-USD')

    # If isOneValid is True, only one asset must be valid for the row to be kept.
    if isOneValid:
        data = data.dropna(axis=1, how='all')
    else:
        data = data.dropna()
    return data

def calculate_benchmark_returns(data):
    ew = EqualWeightedPortfolio(data)
    returns = data.pct_change().dropna()
    weights = ew.calculate_weights()
    cumulative_returns = ew.calculate_cumulative_returns(weights, returns)
    daily_returns = cumulative_returns.pct_change().dropna()
    return daily_returns

def test_portfolios(portfolios, data, benchmark_returns, bootstrap=False):
    for portfolio in portfolios:
        cumulative_returns = portfolio.try_strategy(bootstrap=bootstrap)
        plt.plot(cumulative_returns, label=portfolio.__class__.__name__)

    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

def long_only_usd_portfolios(bootstrap=False):
    data = get_data(inETH=False)
    benchmark_returns = calculate_benchmark_returns(data)

    portfolios = [
        EqualWeightedPortfolio(data, benchmark_returns),
        HRPPortfolio(data, benchmark_returns, distance_metric='euclidean'),
        MinimumVariancePortfolio(data, benchmark_returns),
        RiskParityPortfolio(data, benchmark_returns),
        EqualRiskContributionPortfolio(data, benchmark_returns),
        MaximumDiversificationPortfolio(data, benchmark_returns),
    ]

    print("\033[1;34mTesting Long-Only USD Portfolios\033[0m")
    test_portfolios(portfolios, data, benchmark_returns, bootstrap=bootstrap)

def long_only_eth_portfolios(bootstrap=False):
    data = get_data(inETH=True)
    benchmark_returns = calculate_benchmark_returns(data)

    portfolios = [
        EqualWeightedPortfolio(data, benchmark_returns),
        HRPPortfolio(data, benchmark_returns, distance_metric='euclidean'),
        MinimumVariancePortfolio(data, benchmark_returns),
        RiskParityPortfolio(data, benchmark_returns),
        EqualRiskContributionPortfolio(data, benchmark_returns),
        MaximumDiversificationPortfolio(data, benchmark_returns),
        # MarkowitzPortfolio(data, benchmark_returns),
    ]

    print("\033[1;34mTesting Long-Only ETH Portfolios\033[0m")
    test_portfolios(portfolios, data, benchmark_returns, bootstrap=bootstrap)

def long_short_usd_portfolios(bootstrap=False):
    data = get_data(inETH=False)
    benchmark_returns = calculate_benchmark_returns(data)

    portfolios = [
        EqualWeightedPortfolio(data, benchmark_returns),
        MeanReversionLongShortPortfolio(data, benchmark_returns),
        MomentumLongShortPortfolio(data, benchmark_returns),
        HRPLongShortPortfolio(data, benchmark_returns),
    ]

    print("\033[1;34mTesting Long-Short USD Portfolios\033[0m")
    test_portfolios(portfolios, data, benchmark_returns, bootstrap=bootstrap)

def long_short_eth_portfolios(bootstrap=False):
    data = get_data(inETH=True)
    benchmark_returns = calculate_benchmark_returns(data)

    portfolios = [
        EqualWeightedPortfolio(data, benchmark_returns),
        MeanReversionLongShortPortfolio(data, benchmark_returns),
        MomentumLongShortPortfolio(data, benchmark_returns),
        HRPLongShortPortfolio(data, benchmark_returns),
    ]

    print("\033[1;34mTesting Long-Short ETH Portfolios\033[0m")
    test_portfolios(portfolios, data, benchmark_returns, bootstrap=bootstrap)

def main():
    data = get_data(inETH=True)
    benchmark_returns = calculate_benchmark_returns(data)

    # long_only_usd_portfolios()
    # long_only_eth_portfolios()
    long_short_usd_portfolios()
    long_short_eth_portfolios()

if __name__ == '__main__':
    main()
