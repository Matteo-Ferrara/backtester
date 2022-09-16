import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import src.indicators as indicators
from src.backtesting_engine import Backtester

# Setting up logger
logger = logging.getLogger()
logger_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")
logger_handler.setFormatter(formatter)
logger.addHandler(logger_handler)
logger.setLevel(logging.INFO)

##### Configuration #####

# Investment universe -> [] = All contracts in the folder
selected_markets = []

# Currency choice.
# Set True if you want to use local currency for signals and convert daily change to USD
# Set False if you use data already converted in USD
local_currency = True

# Date settings
starting_date = "1999-01-01"
ending_date = "2051-01-01"

# Portfolio configuration
position_risk = 0.005
initial_equity = 100000000

# Fee / commission configuration
# fee_structure [Mgmt fee, performance fee]
fee = True
fee_structure = [0.02, 0.2]

commission = 10

##### Strategy section #####

# Initializing market and orders df
# all_markets will contain all the daily data to analyze,
# while orders will contain all the signals from the strategy
orders_df = all_markets_df = currencies_df = pd.DataFrame()

# Getting folders paths
root_folder = os.getcwd()
data_folder = "data"
historical_data_folder = "historical_data"
path_to_historical_data = os.path.join(root_folder, data_folder, historical_data_folder)

# Getting list of markets to invest in
if not selected_markets:
    markets_list = [market.split(".")[0] for market in os.listdir(path_to_historical_data)]
else:
    markets_list = selected_markets

# Iterate through markets to generate signals
logger.info("Generating entries and exists...")
for market in tqdm(markets_list):
    # Import data and drop na
    market_data = pd.read_excel(
        os.path.join(root_folder, data_folder, historical_data_folder, market + ".xlsx"),
        usecols=["Dates", "PX_LAST"],
        index_col="Dates",
    ).dropna(how="all")

    # Adding ID column | Used in orders_df to identify ticker
    market_data.insert(0, "Symbol", market)

    # Limit study to certain periods
    if starting_date == "":
        starting_date = None

    if ending_date == "":
        ending_date = None

    market_data = market_data[starting_date:ending_date]

    # Compute indicators
    market_data["fast_ma"] = indicators.simple_moving_average(market_data.PX_LAST, 100)
    market_data["slow_ma"] = indicators.simple_moving_average(market_data.PX_LAST, 200)

    market_data["resistance"] = indicators.local_max(market_data.PX_LAST, 100)
    market_data["support"] = indicators.local_min(market_data.PX_LAST, 100)

    market_data["exit_resistance"] = indicators.local_max(market_data.PX_LAST, 50)
    market_data["exit_support"] = indicators.local_min(market_data.PX_LAST, 50)

    market_data["standard_deviation"] = indicators.standard_deviation(market_data.PX_LAST, 100)

    # vol_parameter is the # of sigmas needed to trigger an exit for volatility
    vol_parameter = 3

    market_data["vol_support"] = (
        indicators.simple_moving_average(market_data.PX_LAST, 20)
        - market_data.standard_deviation * vol_parameter
    )

    market_data["vol_resistance"] = (
        indicators.simple_moving_average(market_data.PX_LAST, 20)
        + market_data.standard_deviation * vol_parameter
    )

    # Set entry / exit rules here
    # Short rules
    short_entry_rule = (market_data.PX_LAST < market_data.support) & (
        market_data.fast_ma < market_data.slow_ma
    )

    short_exit_rule = (market_data.PX_LAST > market_data.exit_resistance) | (
        market_data.PX_LAST < market_data.vol_support
    )

    # Long rules
    long_entry_rule = (market_data.PX_LAST > market_data.resistance) & (
        market_data.fast_ma > market_data.slow_ma
    )

    long_exit_rule = (market_data.PX_LAST < market_data.exit_support) | (
        market_data.PX_LAST > market_data.vol_resistance
    )

    # Compute signals
    market_data["short_signal"] = np.where(
        short_entry_rule, "short", np.where(short_exit_rule, "flat", None)
    )

    market_data["long_signal"] = np.where(
        long_entry_rule, "long", np.where(long_exit_rule, "flat", None)
    )

    # Converting signals to orders
    market_data["short_signal"] = market_data.short_signal.shift(1).ffill()
    market_data["long_signal"] = market_data.long_signal.shift(1).ffill()
    market_data["Order"] = market_data.long_signal + market_data.short_signal
    market_data.Order = (
        market_data.Order.replace(["longflat", "flatflat", "flatshort"], ["long", "flat", "short"])
    )
    market_orders = market_data.where(
        (market_data.Order != market_data.Order.shift()), other=None
    ).dropna()[["Symbol", "Order"]]

    if not market_orders.empty:
        # Skip first line if it's not a new position
        market_orders = market_orders[1:] if market_orders.Order[0] == "flat" else market_orders

        # Merge market orders to all orders df
        orders_df = pd.concat([orders_df, market_orders.reset_index()], ignore_index=True)

    # Create final dataframe for the market | This is the df that will be used in backtesting
    market_data = pd.concat(
        [
            market_orders.Order,
            market_data.PX_LAST,
            market_data.exit_resistance,
            market_data.exit_support,
            pd.Series(np.NaN),
            pd.Series(np.NaN),
            pd.Series(np.NaN),
            pd.Series(np.NaN),
        ],
        ignore_index=True,
        axis=1,
    )

    # Rename columns
    market_data = market_data.rename(
        columns={
            0: f"{market} Order",
            1: f"{market} Close",
            2: f"{market} Resistance",
            3: f"{market} Support",
            4: f"{market} Contracts",
            5: f"{market} Margin",
            6: f"{market} Risk",
            7: f"{market} P/L",
        }
    )

    # Merge market to global dataframe
    all_markets_df = pd.concat([all_markets_df, market_data], axis=1)

# Drop any date with all values NaN and sort by date
all_markets_df = all_markets_df.dropna(how="all").sort_index()

# Initializing columns
for market in markets_list:
    all_markets_df[f"{market} Risk"] = all_markets_df[f"{market} Risk"].fillna(0)
    all_markets_df[f"{market} Contracts"] = all_markets_df[f"{market} Contracts"].fillna(0)
    all_markets_df[f"{market} P/L"] = all_markets_df[f"{market} P/L"].fillna(0)

# Load currencies' spot rates
if local_currency:
    logger.info("Getting currencies' exchange rates...")
    path_to_currencies = os.path.join(root_folder, data_folder, "spot_currencies")
    currencies = [currency.split(".")[0] for currency in os.listdir(path_to_currencies)]

    for currency in tqdm(currencies):
        currency_rates = pd.read_excel(
            os.path.join(root_folder, data_folder, "spot_currencies", currency + ".xlsx"),
            index_col="Dates",
            names=["Dates", currency],
        )
        currencies_df = pd.concat([currencies_df, currency_rates], axis=1)

##### Market Simulation #####
backtester = Backtester(
    all_markets_df,
    initial_equity,
    position_risk,
    markets_list,
    orders_df,
    local_currency,
    currencies_df,
    commission,
    fee,
    fee_structure,
)

logger.info("Backtesting...")
all_markets_df, orders_df = backtester.simulate()

# all_markets_df.to_excel("markets.xlsx")
orders_df.to_excel("orders_summary.xlsx")
all_markets_df.loc[:, ["Margin", "Equity"]].to_excel("portfolio_summary.xlsx")
