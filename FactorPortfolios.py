from Portfolio import Portfolio
import pandas as pd
import numpy as np

class FactorLongShortPortfolio(Portfolio):
    def __init__(self, data, factor_data, benchmark_returns=pd.Series(), rebalance_frequency=30, annual_risk_free_rate=0.05):
        super().__init__(data, benchmark_returns, rebalance_frequency, annual_risk_free_rate)
        self.factor_data = factor_data

    def calculate_weights(self):
        factor_scores = self.factor_data.mean()
        sorted_scores = factor_scores.sort_values()

        half_point = len(sorted_scores) // 2
        weights = pd.Series(0, index=sorted_scores.index)
        weights[sorted_scores.index[:half_point]] = -1 / half_point  # Short low factor score assets
        weights[sorted_scores.index[half_point:]] = 1 / half_point  # Long high factor score assets

        return weights
