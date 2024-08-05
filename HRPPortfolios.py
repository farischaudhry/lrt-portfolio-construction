import yfinance as yf
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist
from Portfolio import Portfolio

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
