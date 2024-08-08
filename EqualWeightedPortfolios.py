from Portfolio import Portfolio
import numpy as np
import pandas as pd

class EqualWeightedPortfolio(Portfolio):
    def calculate_weights(self):
        num_assets = self.data.shape[1]
        return np.array([1 / num_assets] * num_assets)

class RandomWeightedPortfolio(Portfolio):
    def calculate_weights(self):
        weights = np.random.random(len(self.data.columns))
        return weights / weights.sum()
