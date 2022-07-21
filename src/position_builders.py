import sys
import logging
from datetime import timedelta
from math import isnan, ceil


logger = logging.getLogger(__name__)


def get_daily_change(close, close_prev):
    """Function to compute daily change"""
    return (close - close_prev) if not isnan(close - close_prev) else 0


def get_position_points(data, date_idx, market):
    """Function to get points needed to compute positions size"""
    for x in range(0, 10):
        if not isnan(data.iloc[date_idx - x, data.columns.get_loc(f"{market} Close")]):
            return (
                data.iloc[date_idx - x, data.columns.get_loc(f"{market} Close")],
                data.iloc[date_idx - x, data.columns.get_loc(f"{market} Support")],
                data.iloc[date_idx - x, data.columns.get_loc(f"{market} Resistance")],
            )
    logger.error("Error while getting position points.", data.iloc[date_idx])
    sys.exit(1)


def get_mark_to_market_points(data, date_idx, market):
    """Function to take into account weekends and holidays.
    Ex. NKD is closed on 3-4-5 May. This will return Close of 2 and 6"""
    for x in range(1, 10):
        if not isnan(data.iloc[date_idx - x, data.columns.get_loc(f"{market} Close")]):
            return (
                data.iloc[date_idx - x, data.columns.get_loc(f"{market} Close")],
                data.iloc[date_idx, data.columns.get_loc(f"{market} Close")],
            )
    return 0, 0


def get_number_of_contracts(
    data,
    date_idx,
    market,
    position_type,
    updated_equity,
    point_value,
    position_risk,
):
    """Function to compute number of contracts to buy / sell"""
    close, support, resistance = get_position_points(data, date_idx, market)
    # N of contracts is given by (risk factor * equity) / (position risk * point value)

    if position_type == "long":
        if (close - support) != 0 and not isnan(support):
            risk = close - support
        else:
            risk = close * position_risk
        return ceil((position_risk * updated_equity) / (risk * point_value))

    elif position_type == "short":
        if (resistance - close) != 0 and not isnan(resistance):
            risk = resistance - close
        else:
            risk = close * position_risk
        return -ceil((position_risk * updated_equity) / (risk * point_value))

    elif position_type == "flat":
        return 0

    else:
        logger.error(f"Error computing # of contracts. Position Type: {position_type}")
        sys.exit(1)


def get_currency_rate(currencies_df, currency, date):
    """Function to obtain exchange rate in a given day"""
    for x in range(0, 10):
        try:
            rate = currencies_df.at[date - timedelta(days=x), currency]
            if not isnan(rate):
                return rate
        except KeyError:
            continue
