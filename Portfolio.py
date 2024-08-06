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
        ir = excess_returns.mean() / excess_returns.std()
        if np.isnan(ir):
            return 0
        return ir

    def calculate_sharpe_ratio(self, returns):
        excess_returns = returns - self.benchmark_rate
        return excess_returns.mean() / excess_returns.std()

    def calculate_max_drawdown(self, returns):
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return abs(drawdown.min())

    def calmar_ratio(self, returns):
        annualized_return = np.prod(1 + returns) ** (365 / len(returns)) - 1
        return  annualized_return / self.calculate_max_drawdown(returns)

    def print_performance_metrics(self, returns):
        information_ratio = self.calculate_information_ratio(returns)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        max_drawdown = self.calculate_max_drawdown(returns)
        calmar_ratio = self.calmar_ratio(returns)
        print(f"{self.__class__.__name__} Performance Metrics:")
        print(f"Information Ratio: {information_ratio:.3f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Max Drawdown: {max_drawdown:.3f}")
        print(f"Calmar Ratio: {calmar_ratio:.3f}")

    @abstractmethod
    def calculate_weights(self):
        pass

    def try_strategy(self):
        returns = self.data.pct_change().dropna()
        weights = self.calculate_weights()
        cumulative_returns = self.calculate_cumulative_returns(weights, returns)
        daily_returns = cumulative_returns.pct_change().dropna()
        self.print_performance_metrics(daily_returns)
        return cumulative_returns
