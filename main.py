import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.spatial.distance import pdist, squareform
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from datetime import datetime

def get_data(oneValid=True):
    assets = ['EETH-USD', 'RSETH-USD', 'UNIETH-USD', 'PUFETH-USD', 'EZETH-USD', 'RSWETH-USD', 'WEETH-USD']
    end_date = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(assets, start='2020-01-01', end=end_date)['Adj Close']

    if oneValid:
        data = data.dropna(axis=1, how='all')
    else:
        data = data.dropna()
    return data

def calculate_cumulative_returns(weights, returns):
    portfolio_returns = returns.dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    return cumulative_returns

def calculate_sharpe_ratio(returns, risk_free_rate=0.05):
    daily_risk_free_return = (1 + risk_free_rate) ** (1 / 365) - 1
    excess_returns = returns - daily_risk_free_return
    annualized_return = np.prod(1 + excess_returns) ** (365 / len(excess_returns)) - 1
    annualized_std = excess_returns.std() * np.sqrt(365)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std

    return sharpe_ratio

# Equal Weighted Portfolio (Benchmark)
def equal_weighted_portfolio(data):
    returns = data.pct_change().dropna()
    weights = np.array([1/returns.shape[1]] * returns.shape[1])
    return calculate_cumulative_returns(weights, returns)

def hrp_portfolio(data, method='single', distance_metric='euclidean'):
    returns = data.pct_change().dropna()
    corr = returns.corr()
    distance = pdist(1 / corr, distance_metric)
    linkage_matrix = linkage(distance, method=method)
    tree = to_tree(linkage_matrix)

    def get_leaf_nodes(node):
        if node.is_leaf():
            return [node.id]
        return get_leaf_nodes(node.left) + get_leaf_nodes(node.right)

    leaf_nodes = get_leaf_nodes(tree)
    volatilities = returns.std()
    weights = 1 / volatilities[leaf_nodes]
    weights /= weights.sum()

    return calculate_cumulative_returns(weights, returns)

def hrp_long_short_portfolio(data, method='single', distance_metric='euclidean'):
    returns = data.pct_change().dropna()
    corr = returns.corr()
    corr.fillna(0, inplace=True)
    distance = pdist(1 / corr, distance_metric)
    linkage_matrix = linkage(distance, method=method)
    tree = to_tree(linkage_matrix)

    def get_leaf_nodes(node):
        if node.is_leaf():
            return [node.id]
        return get_leaf_nodes(node.left) + get_leaf_nodes(node.right)

    leaf_nodes = get_leaf_nodes(tree)
    volatilities = returns.std()
    inverse_vol_weights = 1 / volatilities
    sorted_vols = inverse_vol_weights.sort_values()
    half_point = len(sorted_vols) // 2
    weights = pd.Series(0, index=sorted_vols.index)
    weights[sorted_vols.index[:half_point]] = -sorted_vols[:half_point] / sorted_vols[:half_point].sum()
    weights[sorted_vols.index[half_point:]] = sorted_vols[half_point:] / sorted_vols[half_point:].sum()

    return calculate_cumulative_returns(weights, returns)

def main():
    data = get_data(oneValid=True)
    cumulative_returns_eq = equal_weighted_portfolio(data)
    cumulative_returns_hrp = hrp_portfolio(data)
    cumulative_returns_hrp_ls = hrp_long_short_portfolio(data)

    daily_returns_eq = cumulative_returns_eq.pct_change().dropna()
    daily_returns_hrp = cumulative_returns_hrp.pct_change().dropna()
    daily_returns_hrp_ls = cumulative_returns_hrp_ls.pct_change().dropna()

    sharpe_eq = calculate_sharpe_ratio(daily_returns_eq)
    sharpe_hrp = calculate_sharpe_ratio(daily_returns_hrp)
    sharpe_hrp_ls = calculate_sharpe_ratio(daily_returns_hrp_ls)

    print(f"Sharpe Ratios:\nEqual Weighted: {sharpe_eq:.2f}\nHRP: {sharpe_hrp:.2f}\nHRP Long-Short: {sharpe_hrp_ls:.2f}")

    plt.figure(figsize=(12, 8))
    plt.plot(cumulative_returns_eq, label='Equal Weighted')
    plt.plot(cumulative_returns_hrp, label='HRP')
    plt.plot(cumulative_returns_hrp_ls, label='HRP Long-Short')
    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
