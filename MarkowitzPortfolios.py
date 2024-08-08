from Portfolio import Portfolio
import riskfolio as rp
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from scipy.optimize import minimize
import numpy as np

class MarkowitzPortfolio(Portfolio):
    def calculate_weights(self):
        returns = self.data.pct_change().dropna()
        mu = expected_returns.mean_historical_return(self.data)
        S = risk_models.sample_cov(self.data)

        # Check if any asset's expected return exceeds the risk-free rate
        risk_free_rate = 0.05
        if not any(mu > risk_free_rate):
            raise ValueError(max(mu))

        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        cleaned_weights = ef.clean_weights()
        return np.array(list(cleaned_weights.values()))

class MinimumVariancePortfolio(Portfolio):
    def calculate_weights(self):
        returns = self.data.pct_change().dropna()
        S = risk_models.sample_cov(self.data)

        ef = EfficientFrontier(None, S)
        weights = ef.min_volatility()
        cleaned_weights = ef.clean_weights()
        return np.array(list(cleaned_weights.values()))

class RiskParityPortfolio(Portfolio):
    def calculate_weights(self):
        returns = self.data.pct_change().dropna()
        volatilities = returns.std()
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        return weights

class EqualRiskContributionPortfolio(Portfolio):
    def calculate_weights(self):
        returns = self.data.pct_change().dropna()
        cov = returns.cov()
        vol = returns.std()
        inv_vol = 1 / vol
        initial_weights = inv_vol / inv_vol.sum()

        def get_risk_contributions(weights):
            portfolio_var = weights.T @ cov @ weights
            marginal_contrib = cov @ weights
            risk_contrib = weights * marginal_contrib / portfolio_var
            return risk_contrib

        def risk_budget_objective(weights, target_risk):
            risk_contrib = get_risk_contributions(weights)
            return np.sum((risk_contrib - target_risk) ** 2)

        target_risk = np.ones(len(initial_weights)) / len(initial_weights)
        bounds = [(0, 1) for _ in range(len(initial_weights))]
        constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        optimized_weights = minimize(risk_budget_objective, initial_weights, args=(target_risk,), bounds=bounds, constraints=constraints).x
        return optimized_weights

class MaximumDiversificationPortfolio(Portfolio):
    def calculate_weights(self):
        returns = self.data.pct_change().dropna()
        cov = returns.cov()
        vol = returns.std()

        def diversification_ratio(weights):
            weighted_vol = np.dot(weights, vol)
            portfolio_vol = np.sqrt(weights.T @ cov @ weights)
            return weighted_vol / portfolio_vol

        num_assets = len(vol)
        initial_weights = np.ones(num_assets) / num_assets
        bounds = [(0, 1) for _ in range(num_assets)]
        constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        optimized_weights = minimize(lambda weights: -diversification_ratio(weights), initial_weights, bounds=bounds, constraints=constraints).x
        return optimized_weights
