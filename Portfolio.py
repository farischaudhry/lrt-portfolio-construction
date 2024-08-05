from abc import ABC, abstractmethod
import numpy as np

class Portfolio(ABC):
    def __init__(self, data):
        self.data = data

    @staticmethod
    def calculate_cumulative_returns(weights, returns):
        portfolio_returns = returns.dot(weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        cumulative_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        cumulative_returns.dropna(inplace=True)
        return cumulative_returns

    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.05):
        daily_risk_free_return = (1 + risk_free_rate) ** (1 / 365) - 1
        excess_returns = returns - daily_risk_free_return
        annualized_return = np.prod(1 + excess_returns) ** (365 / len(excess_returns)) - 1
        annualized_std = excess_returns.std() * np.sqrt(365)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std

        return sharpe_ratio

    @abstractmethod
    def calculate_weights(self):
        pass

    def try_strategy(self):
        returns = self.data.pct_change().dropna()
        weights = self.calculate_weights()
        cumulative_returns = self.calculate_cumulative_returns(weights, returns)
        daily_returns = cumulative_returns.pct_change().dropna()
        sharpe_ratio = self.calculate_sharpe_ratio(daily_returns)
        print(f"{self.__class__.__name__} Sharpe Ratio: {sharpe_ratio:.3f}")
        return cumulative_returns
