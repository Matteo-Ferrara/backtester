import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from scipy.optimize import minimize

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

sde = importr("sde")
stats = importr("stats")

root_folder = os.getcwd()
data_folder = "data"
historical_data_folder = "historical_data_ratio_adjusted"
path_to_historical_data = os.path.join(root_folder, data_folder, historical_data_folder)
markets_list = [market.split(".")[0] for market in os.listdir(path_to_historical_data)]

dt = 1 / 252

# Date settings
starting_date = "" #"1999-01-01"
ending_date = "2022-01-01"


# Explicit formula MLE
def lik(parameters):
    theta_1, theta_2 = parameters
    loglik = norm.logpdf(market_returns, loc=theta_1, scale=theta_2)
    loglik = loglik[~np.isnan(loglik)]
    return -sum(loglik)


params_matrix = pd.DataFrame()

for market in tqdm(markets_list):
    # Import data and drop na
    market_data = pd.read_excel(
        os.path.join(root_folder, data_folder, historical_data_folder, market + ".xlsx"),
        usecols=["Dates", "PX_LAST"],
        index_col="Dates",
    ).dropna(how="all")[:ending_date]

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
    res = minimize(lik, [1, 1], method="Nelder-Mead")
    theta_1 = res.x[0] / dt
    theta_2 = res.x[1] / np.sqrt(dt)

    market_params = pd.DataFrame(
        {
            "market": [market],
            "drift": [theta_1],
            "volatility": [theta_2],
            "cpoint": [market_data.index[0]],
            "last_day": [market_data.index[-1]],
            "last_price": [market_data.PX_LAST[-1]],
        }
    )
    params_matrix = pd.concat([params_matrix, market_params])


params_matrix.to_excel(os.path.join(root_folder, data_folder, "parameters.xlsx"), index=False)
