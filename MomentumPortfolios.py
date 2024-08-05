from Portfolio import Portfolio
import pandas as pd
import numpy as np

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
