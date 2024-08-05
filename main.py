import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, DBSCAN
from pypfopt import EfficientFrontier, risk_models, expected_returns
from datetime import datetime
from abc import ABC, abstractmethod

from Portfolio import Portfolio

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

class EqualWeightedPortfolio(Portfolio):
    def calculate_weights(self):
        num_assets = self.data.shape[1]
        return np.array([1 / num_assets] * num_assets)

class HRPPortfolio(Portfolio):
    def calculate_weights(self):
        returns = self.data.pct_change().dropna()
        corr = returns.corr()
        distance = pdist(1 / corr, 'euclidean')
        linkage_matrix = linkage(distance, method='single')
        tree = to_tree(linkage_matrix)

        def get_leaf_nodes(node):
            if node.is_leaf():
                return [node.id]
            return get_leaf_nodes(node.left) + get_leaf_nodes(node.right)

        leaf_nodes = get_leaf_nodes(tree)
        volatilities = returns.std()
        weights = 1 / volatilities[leaf_nodes]
        weights /= weights.sum()
        return weights

class HRPLongShortPortfolio(Portfolio):
    def calculate_weights(self):
        returns = self.data.pct_change().dropna()
        corr = returns.corr()
        corr.fillna(0, inplace=True)
        distance = pdist(1 / corr, 'euclidean')
        linkage_matrix = linkage(distance, method='single')
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
        return weights

class EqualWeightLongShortPortfolio(Portfolio):
    def calculate_weights(self):
        returns = self.data.pct_change().dropna()
        volatilities = returns.std()
        inverse_vol_weights = 1 / volatilities
        sorted_vols = inverse_vol_weights.sort_values()
        half_point = len(sorted_vols) // 2
        weights = pd.Series(0, index=sorted_vols.index)
        weights[sorted_vols.index[:half_point]] = -sorted_vols[:half_point] / sorted_vols[:half_point].sum()
        weights[sorted_vols.index[half_point:]] = sorted_vols[half_point:] / sorted_vols[half_point:].sum()
        return weights

class MomentumLongShortPortfolio(Portfolio):
    def calculate_weights(self):
        returns = self.data.pct_change().dropna()
        momentum = returns.mean() / returns.std()
        sorted_momentum = momentum.sort_values()
        half_point = len(sorted_momentum) // 2
        weights = pd.Series(0, index=sorted_momentum.index)
        weights[sorted_momentum.index[:half_point]] = -sorted_momentum[:half_point] / sorted_momentum[:half_point].sum()
        weights[sorted_momentum.index[half_point:]] = sorted_momentum[half_point:] / sorted_momentum[half_point:].sum()
        return weights

class MeanReversionLongShortPortfolio(Portfolio):
    def calculate_weights(self):
        returns = self.data.pct_change().dropna()
        mean_reversion = -returns.mean() / returns.std()
        sorted_mean_reversion = mean_reversion.sort_values()
        half_point = len(sorted_mean_reversion) // 2
        weights = pd.Series(0, index=sorted_mean_reversion.index)
        weights[sorted_mean_reversion.index[:half_point]] = -sorted_mean_reversion[:half_point] / sorted_mean_reversion[:half_point].sum()
        weights[sorted_mean_reversion.index[half_point:]] = sorted_mean_reversion[half_point:] / sorted_mean_reversion[half_point:].sum()
        return weights

class KMeansPortfolio(Portfolio):
    def calculate_weights(self):
        returns = self.data.pct_change().dropna()
        corr = returns.corr()
        kmeans = KMeans(n_clusters=3, random_state=42).fit(corr)
        clusters = kmeans.labels_

        cluster_weights = []
        for cluster in range(3):
            cluster_indices = np.where(clusters == cluster)[0]
            volatilities = returns.iloc[:, cluster_indices].std()
            weights = 1 / volatilities
            weights /= weights.sum()
            cluster_weights.append(weights)

        final_weights = np.zeros(len(returns.columns))
        for i, cluster in enumerate(cluster_weights):
            indices = np.where(clusters == i)[0]
            final_weights[indices] = cluster

        return final_weights

class DBSCANPortfolio(Portfolio):
    def calculate_weights(self):
        returns = self.data.pct_change().dropna()
        corr = returns.corr()
        dbscan = DBSCAN(eps=0.5, min_samples=2, metric='precomputed').fit(1 - corr)
        clusters = dbscan.labels_

        unique_clusters = np.unique(clusters)
        cluster_weights = []
        for cluster in unique_clusters:
            if cluster == -1:  # Noise points
                continue
            cluster_indices = np.where(clusters == cluster)[0]
            volatilities = returns.iloc[:, cluster_indices].std()
            weights = 1 / volatilities
            weights /= weights.sum()
            cluster_weights.append(weights)

        final_weights = np.zeros(len(returns.columns))
        for cluster in unique_clusters:
            if cluster == -1:
                continue
            cluster_indices = np.where(clusters == cluster)[0]
            cluster_weight = cluster_weights.pop(0)
            final_weights[cluster_indices] = cluster_weight

        return final_weights

def main():
    data = get_data()

    portfolios = [
        EqualWeightedPortfolio(data),
        HRPPortfolio(data),
        HRPLongShortPortfolio(data),
        EqualWeightLongShortPortfolio(data),
        MomentumLongShortPortfolio(data),
        MeanReversionLongShortPortfolio(data),
        KMeansPortfolio(data),
        DBSCANPortfolio(data)
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
