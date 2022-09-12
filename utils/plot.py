import pandas as pd
import os
from matplotlib import pyplot as plt, dates
import numpy as np

ticker = "ES"

root_folder = os.getcwd()
data_folder = "data"
historical_data_folder = "synthetic_data"
path_to_data = os.path.join(root_folder, data_folder, historical_data_folder, f"{ticker}.xlsx")

df = pd.read_excel(path_to_data, index_col="Dates")[:"2090-01-01"]


def local_max(data, periods):
    """Return maximum value of a specified period"""
    return data.shift().rolling(window=periods).max()


def local_min(data, periods):
    """Return minimum value of a specified period"""
    return data.shift().rolling(window=periods).min()


resistance = local_max(df.PX_LAST, 100)
support = local_min(df.PX_LAST, 50)

fig, ax = plt.subplots(1,1)
plt.grid(color="#2A3459")
plt.ylabel("Price")
plt.xticks(rotation=45)
ax.plot(df, 'black')
# ax.plot(resistance, '--')
# ax.plot(support, '--')
ax.xaxis.set_major_formatter(dates.DateFormatter('%b-%y'))
plt.show()
