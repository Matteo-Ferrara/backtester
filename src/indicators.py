import numpy as np
import pandas as pd


def simple_moving_average(data, periods):
    """Return simple moving average"""
    return data.rolling(window=periods).mean()


def exponential_moving_average(data, periods):
    """Return exponential moving average"""
    return data.ewm(span=periods).mean()


def local_max(data, periods):
    """Return maximum value of a specified period"""
    return data.shift().rolling(window=periods).max()


def local_min(data, periods):
    """Return minimum value of a specified period"""
    return data.shift().rolling(window=periods).min()


def standard_deviation(data, periods):
    """Return rolling standard deviation"""
    return data.shift().rolling(window=periods).std()


def average_true_range(data, periods, style):
    """Return Average True Range (ATR). Can choose between simple and exponential average"""
    high_point = np.where(data["High"] > data["Close"], data["High"], data["Close"])
    low_point = np.where(data["Close"] < data["Low"], data["Close"], data["Low"])
    true_range = pd.Series(high_point - low_point)
    if style == "simple":
        atr = true_range.shift().rolling(window=periods).mean().values
    else:
        atr = true_range.shift().ewm(window=periods).mean().values
    return atr
