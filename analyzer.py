import os
import pandas as pd
import numpy as np
import empyrical as ep
from matplotlib import pyplot as plt, dates
import seaborn as sns

pd.options.mode.chained_assignment = None

# Select benchmark
benchmark = "SPX"

# Getting folders paths
root_folder, data_folder, historical_data_folder = os.getcwd(), "data", "historical_data"
markets_list = [
    market.split(".")[0]
    for market in os.listdir(os.path.join(root_folder, data_folder, historical_data_folder))
]

# Loading orders summary in temporary dataframe
df = pd.read_excel(os.path.join(root_folder, "orders_summary.xlsx"), index_col=1).sort_index()[
    ["Symbol", "Order", "Risk", "Pnl"]
]

# Load portfolio data
portfolio = pd.read_excel(os.path.join(root_folder, "portfolio_summary.xlsx"), index_col=0)

# Load benchmark data
benchmark_path = os.path.join(root_folder, data_folder, "index", f"{benchmark}.xlsx")
benchmark = pd.read_excel(benchmark_path, index_col=0)
portfolio["Benchmark"] = benchmark

# Create dataframe with all orders important data
orders = pd.DataFrame()
for market in markets_list:
    if not df.loc[df.Symbol.astype("str") == market].empty:
        # Remove open positions
        market_orders = (
            df.loc[df.Symbol.astype("str") == market]
            if df.loc[df.Symbol.astype("str") == market, "Order"][-1] == "flat"
            else df.loc[df.Symbol.astype("str") == market][:-1]
        )
        # Shift Pnl
        market_orders.Pnl = market_orders.Pnl.shift(-1)
        # Compute holding time for each position
        market_orders.loc[:, "Holding_time"] = market_orders.index.to_series().diff().shift(-1)

        orders = pd.concat([orders, market_orders])

# Drop any closing order
orders = orders.loc[orders.Order != "flat"].dropna()

## Trades Statistics ##
# Compute R Returns | How much a trade made in unit of risk
orders["R_return"] = orders.Pnl / orders.Risk
orders.to_excel("ordini.xlsx")
cumulative_R_return = orders.R_return.sum()

# AVG holding times
avg_time = orders.Holding_time.mean().round("D")
avg_time_short = orders.loc[orders.Order == "short", "Holding_time"].mean().round("D")
avg_time_long = orders.loc[orders.Order == "long", "Holding_time"].mean().round("D")
avg_time_win = orders.loc[orders.Pnl >= 0, "Holding_time"].mean().round("D")
avg_time_loss = orders.loc[orders.Pnl < 0, "Holding_time"].mean().round("D")

# W/R
number_of_trades = len(orders)
number_of_winning_trades = len(orders.loc[orders.Pnl >= 0])
win_rate = number_of_winning_trades / number_of_trades

number_of_short_trades = len(orders.loc[orders.Order == "short"])
number_of_winning_short_trades = len(orders.loc[(orders.Pnl >= 0) & (orders.Order == "short")])
win_rate_short = number_of_winning_short_trades / number_of_short_trades

number_of_long_trades = len(orders.loc[orders.Order == "long"])
number_of_winning_long_trades = len(orders.loc[(orders.Pnl >= 0) & (orders.Order == "long")])
win_rate_long = number_of_winning_long_trades / number_of_long_trades

# Profit factor
gross_profit = orders.loc[orders.Pnl >= 0, "Pnl"].sum()
gross_losses = abs(orders.loc[orders.Pnl < 0, "Pnl"].sum())
profit_factor = gross_profit / gross_losses

# Distribution of Pnl
avg_pnl = orders.Pnl.mean()
avg_pnl_win = orders.loc[orders.Pnl >= 0, "Pnl"].mean()
avg_pnl_loss = orders.loc[orders.Pnl < 0, "Pnl"].mean()

median_pnl = orders.Pnl.median()
median_pnl_win = orders.loc[orders.Pnl >= 0, "Pnl"].median()
median_pnl_loss = orders.loc[orders.Pnl < 0, "Pnl"].median()

min_pnl = orders.Pnl.min()
max_pnl = orders.Pnl.max()

# Distribution of Returns
avg_return = orders.R_return.mean()
avg_return_win = orders.loc[orders.Pnl >= 0, "R_return"].mean()
avg_return_loss = orders.loc[orders.Pnl < 0, "R_return"].mean()

median_return = orders.R_return.median()
median_return_win = orders.loc[orders.Pnl >= 0, "R_return"].median()
median_return_loss = orders.loc[orders.Pnl < 0, "R_return"].median()

min_return = orders.R_return.min()
max_return = orders.R_return.max()

skew_return = orders.R_return.skew()
kurtosis_return = orders.R_return.kurtosis()

## Market Statistics ##
mkt_returns = orders.groupby("Symbol")["R_return"].sum()
mkt_avg_returns = orders.groupby("Symbol")["R_return"].mean()
mkt_min_returns = orders.groupby("Symbol")["R_return"].min()
mkt_max_returns = orders.groupby("Symbol")["R_return"].max()

number_of_trades_mkt = orders.loc[orders.Pnl >= 0].groupby("Symbol").count()
number_of_winning_trades_mkt = orders.groupby("Symbol").count()
mkt_win_rate = (number_of_winning_trades_mkt / number_of_trades_mkt).Pnl

number_of_long_trades_mkt = (
    orders.loc[(orders.Pnl >= 0) & (orders.Order == "short")].groupby("Symbol").count()
)
number_of_winning_long_trades_mkt = orders.loc[orders.Order == "short"].groupby("Symbol").count()
mkt_short_win_rate = (number_of_winning_long_trades_mkt / number_of_long_trades_mkt).Pnl

number_of_short_trades_mkt = (
    orders.loc[(orders.Pnl >= 0) & (orders.Order == "short")].groupby("Symbol").count()
)
number_of_winning_short_trades_mkt = orders.loc[orders.Order == "short"].groupby("Symbol").count()
mkt_short_win_rate = (number_of_winning_short_trades_mkt / number_of_short_trades_mkt).Pnl

## Portfolio Statistics ##
# Monthly returns
returns_monthly = ep.aggregate_returns(portfolio.Equity.pct_change(), "monthly")
win_rate_monthly = len(returns_monthly.loc[returns_monthly > 0]) / (
    len(returns_monthly.loc[returns_monthly != 0])
)

avg_return_monthly = returns_monthly.mean()
avg_return_win_monthly = returns_monthly[returns_monthly > 0].mean()
avg_return_loss_monthly = returns_monthly[returns_monthly < 0].mean()

median_return_monthly = returns_monthly.median()
median_return_win_monthly = returns_monthly.loc[returns_monthly > 0].median()
median_return_loss_monthly = returns_monthly.loc[returns_monthly < 0].median()

min_return_monthly = returns_monthly.min()
max_return_monthly = returns_monthly.max()

# Yearly returns
returns_yearly = ep.aggregate_returns(portfolio.Equity.pct_change(), "yearly")
win_rate_yearly = len(returns_yearly.loc[returns_yearly > 0]) / (
    len(returns_yearly.loc[returns_yearly != 0])
)

avg_return_yearly = returns_yearly.mean()
avg_return_win_yearly = returns_yearly[returns_yearly > 0].mean()
avg_return_loss_yearly = returns_yearly[returns_yearly < 0].mean()

median_return_yearly = returns_yearly.median()
median_return_win_yearly = returns_yearly.loc[returns_yearly > 0].median()
median_return_loss_yearly = returns_yearly.loc[returns_yearly < 0].median()

min_return_yearly = returns_yearly.min()
max_return_yearly = returns_yearly.max()

# Portfolio Metrics
cumulative_return = (1 + portfolio.Equity.pct_change()).cumprod()
cagr = ep.cagr(portfolio.Equity.pct_change())
max_drawdown = ep.max_drawdown(portfolio.Equity.pct_change())
roll_drawdown = cumulative_return / cumulative_return.cummax() - 1
calmar_ratio = abs(cagr / max_drawdown)
sharpe_ratio = ep.sharpe_ratio(portfolio.Equity.pct_change())
sortino_ratio = ep.sortino_ratio(portfolio.Equity.pct_change())
omega_ratio = ep.omega_ratio(portfolio.Equity.pct_change())
tail_ratio = ep.tail_ratio(portfolio.Equity.pct_change())
annual_volatility = ep.annual_volatility(portfolio.Equity.pct_change())

daily_avg_return_equity = portfolio.Equity.pct_change().mean()
daily_avg_pnl_equity = portfolio.Equity.diff().mean()
daily_standard_deviation_equity = portfolio.Equity.pct_change().std()
daily_skew_equity = portfolio.Equity.pct_change().skew()
daily_kurtosis_equity = portfolio.Equity.pct_change().kurtosis()

alpha, beta = ep.alpha_beta(portfolio.Equity.pct_change(), portfolio.Benchmark.pct_change())
rolling_beta = ep.roll_beta(
    portfolio.Equity.pct_change(), portfolio.Benchmark.pct_change(), window=252
)

correlation = portfolio.Equity.pct_change().corr(portfolio.Benchmark.pct_change())
roll_correlation = (
    portfolio.Equity.pct_change().rolling(252).corr(portfolio.Benchmark.pct_change())
)

length_drawdown = roll_drawdown.loc[roll_drawdown == 0].index.to_series().diff()
# print(length_drawdown.loc[length_drawdown != "1 days"].describe())

# Margin to Equity
margin_to_equity = portfolio.Margin / portfolio.Equity
avg_margin_to_equity = margin_to_equity.mean()
max_margin_to_equity = margin_to_equity.max()


### Print statistics
# print(f'Cumulative R return: {cumulative_R_return}')

# print(f"The average holding time is: {avg_time}")
# print(f'The average holding time for long positions is: {avg_time_long}')
# print(f'The average holding time for short positions is: {avg_time_short}')
# print(f"The average holding time for winning positions is: {avg_time_win}")
# print(f"The average holding time for losing positions is: {avg_time_loss}")


# print(f'The win rate is: {win_rate:.4f}')
# print(f'The short leg win rate is: {win_rate_short:.4f}')
# print(f'The long leg win rate is: {win_rate_long:.4f}')

# print(f'The profit factor is: {profit_factor}')

# print(f'The average PnL of a win is: {win_avg_pnl}')
# print(f'The average Pnl of a loss is: {loss_avg_pnl}')

# print(f'The average return of a win is: {avg_return_win}')
# print(f'The average return of a loss is: {avg_return_loss}')

print(f"The CAGR is: {cagr:.4f}")
print(f"The annual volatility is: {annual_volatility:.4f}")
print(f"The max drawdown is: {max_drawdown:.4f}")
print(f"The sharpe ratio is: {sharpe_ratio:.4f}")
print(f"The sortino ratio is: {sortino_ratio:.4f}")
print(f"Calmar ratio: {calmar_ratio:.4f}")

print(f"Correlation: {correlation:.4f}")
print(f"Alpha, Beta: {alpha:.4f} {beta:.4f}")

### Plots ###

# Plot histogram of returns
# sns.histplot(orders.R_return)
# plt.title(f"Return's distribution")
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(margin_to_equity, color='k')
# ax.set_title("Margin to equity")
# ax.xaxis.set_major_formatter(dates.DateFormatter('%b-%y'))
# plt.xticks(rotation=45)
# plt.show()

# fig, ax = plt.subplots()
# ax.set_title(f"Rolling beta")
# ax.plot(rolling_beta, color='k')
# ax.xaxis.set_major_formatter(dates.DateFormatter('%b-%y'))
# plt.xticks(rotation=45)
# plt.show()

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1, 1]})
# ax1.plot(portfolio.Equity / portfolio.Equity[0], color='k')
# ax1.set_ylabel('Return')
# ax1.set_title("Equity Curve")
# ax1.set_xticklabels([])
# ax1.set_xticks([])
# ax1.set_ylim(bottom=1)
# ax2.plot(roll_drawdown, color='k')
# ax2.fill_between(roll_drawdown.index, 0, roll_drawdown, color='r')
# ax2.set_ylabel('Drawdown')
# ax2.set_xticklabels([])
# ax2.set_xticks([])
# ax2.set_ylim(ymax=0)
# ax3.plot(roll_correlation, color='k')
# ax3.set_ylabel('Correlation')
# ax3.axhline(y=0, color='r', linestyle='--')
# ax3.xaxis.set_major_formatter(dates.DateFormatter('%b-%y'))
# plt.show()

# aggregate = 0.5 * (1 + np.cumsum(portfolio.Equity.pct_change())) + 0.5 * (1 + np.cumsum(portfolio.Benchmark.pct_change()))
# print(ep.sharpe_ratio(portfolio.Benchmark.pct_change()))
# print(ep.sharpe_ratio(aggregate.pct_change()))
