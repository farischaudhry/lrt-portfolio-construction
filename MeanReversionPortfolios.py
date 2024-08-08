from Portfolio import Portfolio
import pandas as pd
import numpy as np

class MeanReversionLongShortPortfolio(Portfolio):
    def calculate_weights(self):
        returns = self.data.pct_change().dropna()
        mean_reversion_scores = returns.apply(lambda x: (x - x.mean()) / x.std())
        sorted_scores = mean_reversion_scores.mean().sort_values()

        half_point = len(sorted_scores) // 2
        weights = pd.Series(0, index=sorted_scores.index)
        weights[sorted_scores.index[:half_point]] = 1 / half_point  # Long losers
        weights[sorted_scores.index[half_point:]] = -1 / half_point  # Short winners

        return weights
