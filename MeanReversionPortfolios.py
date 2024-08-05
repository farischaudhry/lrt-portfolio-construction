from Portfolio import Portfolio
import pandas as pd
import numpy as np

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
