from Portfolio import Portfolio
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
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
