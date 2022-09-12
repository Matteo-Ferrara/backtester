import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

root_folder, data_folder, historical_data_folder = os.getcwd(), "data", "synthetic_data"

markets_list = [
    market.split(".")[0]
    for market in os.listdir(os.path.join(root_folder, data_folder, historical_data_folder))
    
]

close_matrix = pd.DataFrame()
for market in markets_list:
    data = pd.read_excel(
        os.path.join(root_folder, data_folder, historical_data_folder, market + ".xlsx"),
        index_col="Dates",
    )[["PX_LAST"]].dropna()
    data.index = pd.to_datetime(data.index)
    # data = data["2000-01-01":"2021-12-31"]
    

    close_matrix = pd.concat([close_matrix, data["PX_LAST"].rename("{}".format(market))], axis=1).sort_index()


correlation = np.log(close_matrix).diff().corr()

mask = np.triu(np.ones_like(correlation, dtype=bool))

sn.heatmap(
    np.log(close_matrix).diff().corr(),
    mask=mask,
    annot=True,
    xticklabels=True,
    linewidths=0.5,
    yticklabels=True,
    fmt=".2f",
    vmin=-1,
    vmax=1,
    center=0,
)

#correlation.to_csv(f"correlation/Global.csv")
plt.show()