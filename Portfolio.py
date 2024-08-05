from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class Portfolio(ABC):
    def __init__(self, data, benchmark_returns=pd.Series()):
        self.data = data
        self.benchmark_returns = benchmark_returns
        self.benchmark_rate = benchmark_returns.mean()

    @staticmethod
    def calculate_cumulative_returns(weights, returns):
        portfolio_returns = returns.dot(weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        cumulative_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        cumulative_returns.dropna(inplace=True)
        return cumulative_returns

    def calculate_information_ratio(self, returns):
        excess_returns = returns - self.benchmark_returns
        return excess_returns.mean() / excess_returns.std()

    @abstractmethod
    def calculate_weights(self):
        pass

    def try_strategy(self):
        returns = self.data.pct_change().dropna()
        weights = self.calculate_weights()
        cumulative_returns = self.calculate_cumulative_returns(weights, returns)
        daily_returns = cumulative_returns.pct_change().dropna()
        information_ratio = self.calculate_information_ratio(daily_returns)
        print(f"{self.__class__.__name__} Information Ratio: {information_ratio:.3f}")
        return cumulative_returns
