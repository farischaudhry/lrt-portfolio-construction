from Portfolio import Portfolio
import numpy as np
import pandas as pd

class EqualWeightedPortfolio(Portfolio):
    def calculate_weights(self):
        num_assets = self.data.shape[1]
        return np.array([1 / num_assets] * num_assets)

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
