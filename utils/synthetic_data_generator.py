import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

T = 35
dt = 1 / 252
grid_points = int(T * 252)
paths = 1

# Random Generator instance
rng = np.random.default_rng()

root_folder = os.getcwd()
data_folder = "data"
historical_data_folder = "historical_data_ratio_adjusted"
synthetic_data_folder = 'synthetic_data'
path_to_historical_data = os.path.join(root_folder, data_folder, historical_data_folder)

param_matrix = pd.read_excel(
    os.path.join(root_folder, data_folder, "parameters.xlsx")
)

for idx, market in enumerate(tqdm(param_matrix.market)):
    theta_1, theta_2 = param_matrix.drift[idx], param_matrix.volatility[idx]
    wiener_process = rng.standard_normal([paths, grid_points]) * np.sqrt(dt)
    drift_component = theta_1 * dt
    diffusive_component = theta_2 * wiener_process
    returns = np.cumsum(drift_component + diffusive_component, axis=1)
    simulated_data = param_matrix.last_price[idx] * np.exp(returns)
    selected_simulated_path = np.sort(simulated_data.T).T[int(paths / 2)]
    last_day = param_matrix.last_day[idx]
    simulated_index = pd.date_range(start=last_day + timedelta(days=1), end=last_day + timedelta(days=len(selected_simulated_path + 1)))

    # data = pd.concat([market_data, pd.DataFrame(selected_simulated_path, index=simulated_index, columns=['PX_LAST'])])
    data = pd.DataFrame(selected_simulated_path, index=simulated_index, columns=['PX_LAST'])
    data.to_excel(os.path.join(root_folder, data_folder, synthetic_data_folder, f'{market}.xlsx'), index_label='Dates')