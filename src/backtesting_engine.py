import logging
import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.position_builders import (
    get_mark_to_market_points,
    get_number_of_contracts,
    get_daily_change,
    get_position_points,
    get_currency_rate,
)


class Backtester:
    """
    Class used to simulate trading strategies
    """

    def __init__(
        self,
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
    ):
        """
        :param pandas.DataFrame all_markets_df: df including historical data
        :param float initial_equity: Initial equity level
        :param float position_risk: % risk level for each new position
        :param list markets_list: List of all available markets
        :param pandas.DataFrame orders_df: df summarizing all trading orders
        :param bool local_currency: Attribute to signal if the conversion in USD is needed
        :param pandas.DataFrame currencies_df: df including exchange rates
        :param float commission: Commission per trade
        :param bool fee: Attribute to include fees
        :param list fee_structure: List that include fee structure. First value is mgmt fee, the second is performance fee / carry
        """
        self.logger = logging.getLogger(__name__)
        self.data = all_markets_df.copy()  # To avoid fragmentation of DF
        self.position_risk = position_risk
        self.markets_list = markets_list
        self.orders_df = orders_df
        self.local_currency = local_currency
        self.currencies_df = currencies_df
        self.commission = commission * 2
        self.fee = fee
        self.fee_structure = fee_structure

        # Initialize columns
        self.data["Margin"] = 0.0
        self.data["Equity"] = self.data["Watermark"] = initial_equity
        self.orders_df["Risk"] = 0.0

        # Get portfolio's equity, margin and watermark indices
        self.general_equity_idx = self.data.columns.get_loc("Equity")
        self.general_margin_idx = self.data.columns.get_loc("Margin")
        self.watermark_idx = self.data.columns.get_loc("Watermark")

        # Load file with Futures contracts specifications
        self.specifications = pd.read_excel(
            os.path.join(os.getcwd(), "data", "contracts_details.xlsx"),
            usecols=["Symbol", "Currency", "Point_Value", "Margin"],
        )

    @lru_cache
    def simulate(self):
        # Iterate through each day
        for day_data in tqdm(self.data.itertuples(), total=self.data.shape[0]):
            # Get date
            date = day_data[0].to_pydatetime()
            # Get index for selected day
            date_idx = self.data.index.get_indexer([date])[0]
            # Initialize mark to market change for the day
            marked_to_market = 0

            # Iterate through each market
            for market in self.markets_list:

                # Get indices for accessing DF
                order_idx = self.data.columns.get_loc(f"{market} Order")
                contract_idx = self.data.columns.get_loc(f"{market} Contracts")
                risk_idx = self.data.columns.get_loc(f"{market} Risk")
                margin_idx = self.data.columns.get_loc(f"{market} Margin")
                pnl_idx = self.data.columns.get_loc(f"{market} P/L")

                market_name = market.split('_')[0]
                # Get position of selected market contract specifications
                market_specifications = self.specifications.Symbol.values.astype("str") == market_name
                # Get point value
                point_value = self.specifications[market_specifications].Point_Value.values[0]
                # Get margin requirement
                margin_requirement = self.specifications[market_specifications].Margin.values[0]

                # Check if there's new position change
                if self.data.iat[date_idx - 1, order_idx] is not np.NaN and date_idx != 0:
                    self.new_order(
                        date_idx,
                        contract_idx,
                        risk_idx,
                        pnl_idx,
                        market,
                        self.data.iat[date_idx - 1, order_idx],
                        point_value,
                    )

                    # Commission per roundtrip | We anticipate payment
                    if self.data.iat[date_idx, contract_idx] != 0:
                        marked_to_market -= self.data.iat[date_idx, contract_idx] * self.commission

                else:
                    # Copy previous day # of contracts
                    self.data.iat[date_idx, contract_idx] = self.data.iat[
                        date_idx - 1, contract_idx
                    ]

                    # Copy previous day Risk
                    self.data.iat[date_idx, risk_idx] = self.data.iat[date_idx - 1, risk_idx]

                # Compute daily change
                daily_change = self.mark_to_market(
                    date_idx,
                    order_idx,
                    contract_idx,
                    pnl_idx,
                    point_value,
                    market,
                )

                # Convert to USD if foreign
                if self.local_currency:
                    daily_change = self.convert_to_usd(
                        "change",
                        date_idx,
                        date,
                        margin_idx,
                        market,
                        daily_change,
                    )

                # Round market daily P/L and add it MTM for the day
                marked_to_market += round(daily_change, 4)

                # Compute margins requirement for # of contracts
                self.data.iat[date_idx, margin_idx] = (
                    abs(self.data.iat[date_idx, contract_idx]) * margin_requirement
                )

                # Convert margin to USD if foreign
                if self.local_currency:
                    self.convert_to_usd("margin", date_idx, date, margin_idx, market, 0)

                # Add position margin to total margin requirement for the day
                self.data.iat[date_idx, self.general_margin_idx] += self.data.iat[
                    date_idx, margin_idx
                ]

            # Update equity level
            self.data.iat[date_idx, self.general_equity_idx] = (
                marked_to_market + self.data.iat[date_idx - 1, self.general_equity_idx]
            )

            # Set NAV watermark
            self.data.iat[date_idx, self.watermark_idx] = np.max(
                self.data.iloc[:date_idx, self.general_equity_idx]
            )

            # Remove fees
            if self.fee and date_idx != 0:
                self.compute_fees(date_idx, self.watermark_idx, self.general_equity_idx)

        return self.data, self.orders_df

    def new_order(
        self,
        date_idx,
        contract_idx,
        risk_idx,
        pnl_idx,
        market,
        order,
        point_value,
    ):
        """
        Update # of contracts and compute position risk

        :param int date_idx: Selected date's row index
        :param int contract_idx: Contracts column index
        :param int risk_idx: Risk column index
        :param int pnl_idx: P/L column index
        :param str market: Name of selected market
        :param str order: Order type i.e. long, short, flat
        :param int point_value: Point value
        """

        # Update # of contracts
        self.data.iat[date_idx, contract_idx] = get_number_of_contracts(
            self.data,
            date_idx,
            market,
            order,
            self.data.iat[date_idx - 1, self.general_equity_idx],
            point_value,
            self.position_risk,
        )

        # Get index for specified order in orders_df
        order_idx = (self.orders_df["Symbol"] == market) & (
            self.orders_df.Dates == self.data.index[date_idx - 1]
        )

        # Save trade pnl
        self.orders_df.loc[order_idx, "Pnl"] = self.data.iat[date_idx - 1, pnl_idx]

        # Compute starting risk
        close, support, resistance = get_position_points(self.data, date_idx, market)
        if order:
            if order == "flat":
                risk_per_contract = 0
            elif order == "long":
                risk_per_contract = close - support
            elif order == "short":
                risk_per_contract = resistance - close
            else:
                self.logger.error("Couldn't recognise order type:", order)
                sys.exit(1)
            # Save starting risk in orders df
            self.orders_df.loc[order_idx, "Risk"] = self.data.iat[date_idx, risk_idx] = abs(
                (self.data.iat[date_idx, contract_idx] * risk_per_contract * point_value)
            )

        return

    def mark_to_market(
        self,
        date_idx,
        order_idx,
        contract_idx,
        pnl_idx,
        point_value,
        market,
    ):
        """
        Mark to market

        :param int date_idx: Selected date's row index
        :param int order_idx: Order column index
        :param int contract_idx: Contracts column index
        :param int pnl_idx: P/L column index
        :param int point_value: Point value
        :param str market: Name of selected market
        """

        # Get Close and Previous Close to mark to market
        close_prev, close = get_mark_to_market_points(self.data, date_idx, market)

        # Compute daily change
        daily_change = (
            get_daily_change(close, close_prev)
            * point_value
            * self.data.iat[date_idx, contract_idx]
        )

        # If we closed position initialize P/L
        if self.data.iat[date_idx - 1, order_idx] == "flat":
            self.data.iat[date_idx, pnl_idx] = 0

        elif self.data.iat[date_idx - 1, order_idx] is not np.NaN:
            self.data.iat[date_idx, pnl_idx] = round(daily_change, 4)

        # If we are in a position keep adding daily change
        else:
            self.data.iat[date_idx, pnl_idx] = (
                round(daily_change, 4) + self.data.iat[date_idx - 1, pnl_idx]
            )

        return daily_change

    def convert_to_usd(self, type, date_idx, date, margin_idx, market, daily_change):
        """
        Convert value to USD

        :param str type: Type of value to convert. margin or change
        :param int date_idx: Selected date's row index
        :param datetime.date date: Selected date
        :param int margin_idx: Margin column index
        :param str market: Selected market
        :param float daily_change: Daily change
        """
        # Load exchange rate
        currency = self.specifications.loc[self.specifications.Symbol.astype("str") == market]
        currency = currency.Currency.values[0]
        rate = get_currency_rate(self.currencies_df, currency, date)
        if type == "margin":
            if currency != "USD":
                self.data.iat[date_idx, margin_idx] *= round(rate, 6)
            return

        elif type == "change":
            if currency != "USD" and daily_change != 0:
                daily_change = daily_change * round(rate, 6)
            return daily_change

        else:
            self.logger.error(f"Error emerged when converting to USD. Type: {type}")
            sys.exit(1)

    def compute_fees(self, date_idx, watermark_idx, general_equity_idx):
        """
        Compute management and incentive fees

        :param int date_idx: Selected date's row index
        :param int watermark_idx: Watermark column index
        :param int general_equity_idx: Equity column index
        """
        profit = max(
            0, self.data.iat[date_idx, general_equity_idx] - self.data.iat[date_idx, watermark_idx]
        )
        self.data.iat[date_idx, general_equity_idx] -= (
            self.data.iat[date_idx, general_equity_idx] * self.fee_structure[0] / 365
            + profit * self.fee_structure[1]
        )
        return
