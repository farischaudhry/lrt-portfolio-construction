from Portfolio import Portfolio
import numpy as np

class DeltaNeutralPortfolio(Portfolio):
    def calculate_weights(self):
        num_assets = self.data.shape[1] - 1  # Excluding ETH-USD
        lrt_weights = np.array([1 / num_assets] * num_assets) / 2  # Dividing weights by 2
        eth_weight = -0.5  # Short position in ETH
        return np.append(lrt_weights, eth_weight)

    def calculate_cumulative_returns(self, weights, returns):
        # Separate LRT and ETH returns
        lrt_returns = returns.iloc[:, :-1]
        eth_returns = returns.iloc[:, -1]

        # Calculate portfolio returns
        lrt_portfolio_returns = lrt_returns.dot(weights[:-1])
        eth_short_returns = weights[-1] * eth_returns

        # Delta-neutral portfolio returns
        portfolio_returns = lrt_portfolio_returns + eth_short_returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        cumulative_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        cumulative_returns.dropna(inplace=True)
        return cumulative_returns
