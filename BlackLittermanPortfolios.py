from Portfolio import Portfolio
import riskfolio as rp
import pandas as pd

# class BlackLittermanPortfolio(Portfolio):
#     def calculate_weights(self):
#         returns = self.data.pct_change().dropna()
#         port = rp.Portfolio(returns)

#         method_mu = 'hist'
#         method_cov = 'hist'

#         port.assets_stats(method_mu=method_mu, method_cov=method_cov)

#         # Ensure that covariance matrix is numeric
#         port.cov = port.cov.astype(float)

#         port.alpha = 0
#         model = 'BL'  # Black Litterman
#         rm = 'MV'  # Risk measure used, this time will be variance
#         obj = 'Sharpe'  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
#         hist = False  # Use historical scenarios for risk measures that depend on scenarios
#         rf = 0  # Risk-free rate

#         weights = port.optimization(model=model, rm=rm, obj=obj, rf=rf, hist=hist)
#         return weights['weights'].to_numpy()
