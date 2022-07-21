import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from datetime import timedelta
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

sde = importr('sde')
stats = importr('stats')

T = 150
dt = 1 / 252
grid_points = int(T * 252)
paths = 10

# Random Generator instance
rng = np.random.default_rng()

root_folder = os.getcwd()
data_folder = "data"
historical_data_folder = "historical_data_ratio_adjusted"
synthetic_data_folder = 'synthetic_data'
path_to_historical_data = os.path.join(root_folder, data_folder, historical_data_folder)
markets_list = [market.split(".")[0] for market in os.listdir(path_to_historical_data)]


# Explicit formula MLE
def lik(parameters):
    theta_1, theta_2 = parameters
    loglik = norm.logpdf(market_returns, loc=theta_1, scale=theta_2)
    loglik = loglik[~np.isnan(loglik)]
    return -sum(loglik)


column_list = ['Market', 'Drift', 'Volatility', 'Cpoint']
params_matrix = pd.DataFrame(columns=column_list)

for market in tqdm(markets_list):
    # Import data and drop na
    market_data = pd.read_excel(
        os.path.join(root_folder, data_folder, historical_data_folder, market + ".xlsx"),
        usecols=["Dates", "PX_LAST"],
        index_col="Dates",
    ).dropna(how="all")['1999-01-01':'2020-01-01']

    # Convert data from Pandas.DataFrame to R-format
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(market_data)

    # Convert vector to time series
    rts_data = stats.ts(r_data)

    # Change-point analysis
    out_cpoint = sde.cpoint(rts_data)

    # Extract $tau1 (index of change-point)
    cpoint = int(np.array(out_cpoint[1]))

    market_data = market_data[cpoint:]

    market_returns = np.log(market_data.PX_LAST).diff()
    market_returns = market_returns[~np.isnan(market_returns)]
    res = minimize(lik, [1, 1], method='Nelder-Mead')

    # Fit function
    theta_1 = (res.x[0] / dt)
    theta_2 = (res.x[1] / np.sqrt(dt))
    market_params = pd.DataFrame({'Market': [market], 'Drift': [theta_1], 'Volatility': [theta_2], 'Cpoint': [market_data.index[0]]})
    params_matrix = pd.concat([params_matrix, market_params])

    wiener_process = rng.standard_normal([paths, grid_points]) * np.sqrt(dt)
    drift_component = theta_1 * dt
    diffusive_component = theta_2 * wiener_process
    returns = np.cumsum(drift_component + diffusive_component, axis=1)
    simulated_data = market_data.PX_LAST[-1] * np.exp(returns)
    selected_simulated_path = np.sort(simulated_data.T).T[int(paths / 2)]
    # selected_simulated_path = simulated_data.T
    last_day = market_data.index[-1]
    simulated_index = pd.date_range(start=last_day + timedelta(days=1), end=last_day + timedelta(days=len(selected_simulated_path + 1)))

    # data = pd.concat([market_data, pd.DataFrame(selected_simulated_path, index=simulated_index, columns=['PX_LAST'])])
    data = pd.DataFrame(selected_simulated_path, index=simulated_index, columns=['PX_LAST'])
    data.to_excel(os.path.join(root_folder, data_folder, synthetic_data_folder, f'{market}.xlsx'), index_label='Dates')


params_matrix.to_excel(os.path.join(root_folder, data_folder, 'parameters.xlsx'), index=False)
