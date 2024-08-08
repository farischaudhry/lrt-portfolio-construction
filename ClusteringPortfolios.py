import numpy as np
from Portfolio import Portfolio
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd

class KMeansPortfolio(Portfolio):
    def __init__(self, data, benchmark_returns, n_clusters=3):
        self.n_clusters = n_clusters
        super().__init__(data, benchmark_returns)

    def calculate_weights(self):
        returns = self.data.pct_change().dropna()
        corr = returns.corr()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto').fit(corr)
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
