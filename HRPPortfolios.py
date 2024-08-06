import yfinance as yf
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import cosine
from sklearn.feature_selection import mutual_info_classif
from Portfolio import Portfolio

class HRPPortfolio(Portfolio):
    def __init__(self, data, benchmark_returns, distance_metric='euclidean'):
        super().__init__(data, benchmark_returns)
        self.distance_metric = distance_metric

    def calculate_weights(self):
        returns = self.data.pct_change().dropna()
        corr = returns.corr()

        if self.distance_metric == 'mutual_info':
            # Discretize the continuous returns into bins
            bins = np.apply_along_axis(lambda x: pd.cut(x, bins=5, labels=False), axis=0, arr=returns)
            distance = pdist(bins.T, metric=lambda x, y: mutual_info_score(x, y))
        elif self.distance_metric == 'cosine':
            distance = pdist(returns.T, metric='cosine')
        else:  # Default to euclidean
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
