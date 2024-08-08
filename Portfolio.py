from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.utils import resample

class Portfolio(ABC):
    def __init__(self, data, benchmark_returns=pd.Series(), rebalance_frequency=30, annual_risk_free_rate=0.05):
        self.data = data
        self.benchmark_returns = benchmark_returns
        self.daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1 / 365) - 1
        self.rebalance_frequency = rebalance_frequency

    @staticmethod
    def calculate_cumulative_returns(weights, returns):
        weighted_returns = (returns * weights).sum(axis=1)
        cumulative_returns = (1 + weighted_returns).cumprod()
        cumulative_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        cumulative_returns.dropna(inplace=True)
        return cumulative_returns

    def rebalance(self, returns):
        weights_list = []
        rebalance_dates = []
        for start, end in self.get_rebalance_intervals(returns):
            interval_weights = self.calculate_weights()
            weights_list.append(interval_weights)
            rebalance_dates.append(returns.index[start])

        # Create a DataFrame of weights with appropriate date index
        weights_df = pd.DataFrame(weights_list, index=rebalance_dates, columns=returns.columns)
        return weights_df.reindex(returns.index, method='ffill').fillna(method='ffill')

    def get_rebalance_intervals(self, returns):
        if self.rebalance_frequency == None:
            return [(0, len(returns))]
        else:
            return [(i, i + 1) for i in range(0, len(returns), self.rebalance_frequency)]


    def calculate_information_ratio(self, returns):
        excess_returns = returns - self.benchmark_returns
        ir = excess_returns.mean() / excess_returns.std()
        if np.isnan(ir):
            return 0
        return ir * np.sqrt(365)

    def calculate_sharpe_ratio(self, returns):
        excess_returns = returns - self.daily_risk_free_rate
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(365)

    def calculate_max_drawdown(self, returns):
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return abs(drawdown.min())

    def calculate_calmar_ratio(self, returns):
        annualized_return = np.prod(1 + returns) ** (365 / len(returns)) - 1
        return  annualized_return / self.calculate_max_drawdown(returns)

    def calculate_performance_metrics(self, returns):
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        information_ratio = self.calculate_information_ratio(returns)
        max_drawdown = self.calculate_max_drawdown(returns)
        calmar_ratio = self.calculate_calmar_ratio(returns)
        annualized_return = np.prod(1 + returns) ** (365 / len(returns)) - 1
        annualized_volatility = returns.std() * np.sqrt(365)

        return {
            "Sharpe Ratio": sharpe_ratio,
            "Information Ratio": information_ratio,
            "Max Drawdown": max_drawdown,
            "Calmar Ratio": calmar_ratio,
            "Volatility": annualized_volatility,
        }

    def performance_attribution(self, returns, weights):
        trading_days_per_year = 365

        # Annualizing the returns and benchmark returns
        annualized_returns = ((1 + returns).prod() ** (trading_days_per_year / len(returns))) - 1
        annualized_benchmark_return = ((1 + self.benchmark_returns).prod() ** (trading_days_per_year / len(self.benchmark_returns))) - 1

        # Calculate the effects
        allocation_effect = (weights.mean(axis=0) - 1/len(weights.columns)) * annualized_benchmark_return
        selection_effect = (annualized_returns - annualized_benchmark_return) * 1/len(weights.columns)
        interaction_effect = (weights.mean(axis=0) - 1/len(weights.columns)) * (annualized_returns - annualized_benchmark_return)

        total_allocation_effect = allocation_effect.sum()
        total_selection_effect = selection_effect.sum()
        total_interaction_effect = interaction_effect.sum()

        total_effect = total_allocation_effect + total_selection_effect + total_interaction_effect

        return {
            "Portfolio Returns (Annualized)": annualized_returns,
            "Benchmark Returns (Annualized)": annualized_benchmark_return,
            "Difference in Returns": annualized_returns - annualized_benchmark_return,
            "Allocation Effect": total_allocation_effect,
            "Selection Effect": total_selection_effect,
            "Interaction Effect": total_interaction_effect,
        }

    def bootstrap_performance_metrics(self, returns, block_size=20, num_bootstrap=1000):
        def block_bootstrap(returns, block_size):
            num_blocks = int(np.ceil(len(returns) / block_size))
            indices = np.arange(len(returns))
            sampled_indices = np.concatenate([resample(indices, n_samples=block_size, replace=False) for _ in range(num_blocks)])
            return returns.iloc[sampled_indices[:len(returns)]]

        metrics = []
        for _ in range(num_bootstrap):
            sample_returns = block_bootstrap(returns, block_size)
            metrics.append(self.calculate_performance_metrics(sample_returns))
        return pd.DataFrame(metrics).describe()

    def print_performance_metrics(self, returns):
        information_ratio = self.calculate_information_ratio(returns)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        max_drawdown = self.calculate_max_drawdown(returns)
        calmar_ratio = self.calculate_calmar_ratio(returns)
        print(f"Information Ratio: {information_ratio:.3f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Max Drawdown: {max_drawdown:.3f}")
        print(f"Calmar Ratio: {calmar_ratio:.3f}")

    @abstractmethod
    def calculate_weights(self):
        pass

    def try_strategy(self, bootstrap=False):
        returns = self.data.pct_change().dropna()
        weights = self.rebalance(returns)
        cumulative_returns = self.calculate_cumulative_returns(weights, returns)
        daily_returns = cumulative_returns.pct_change().dropna()

        print(f"\033[1;31m{self.__class__.__name__} Performance Metrics:\033[0m")
        for metric, value in self.calculate_performance_metrics(daily_returns).items():
            print(f"{metric}: {value:.5f}")
        for metric, value in self.performance_attribution(daily_returns, weights).items():
            print(f"{metric}: {value}")

        if bootstrap:
            print("Bootstrap Performance Metrics:")
            print(self.bootstrap_performance_metrics(daily_returns))

        print("-" * 80)

        return cumulative_returns
