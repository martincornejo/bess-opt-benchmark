import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_trading_power(res, price):
    fig, ax = plt.subplots(2, 2, height_ratios=[1, 3], width_ratios=[3, 1])

    df = res.join(price)
    power = df["power_sim"] / 100e3
    price = df["Intraday Continuous 15 minutes ID1-Price"]

    idx = power > 0
    # ax.scatter(price.loc[idx], power.loc[idx])
    ax[1, 0].scatter(power.loc[idx], price.loc[idx], alpha=0.5)
    ax[0, 0].hist(power.loc[idx], alpha=0.5, label="Charge")
    ax[1, 1].hist(price.loc[idx], orientation="horizontal", alpha=0.5)

    idx = power < 0
    # ax.scatter(price.loc[idx], power.loc[idx])
    ax[1, 0].scatter(power.loc[idx], price.loc[idx], alpha=0.5)
    ax[0, 0].hist(power.loc[idx], alpha=0.5, label="Discharge")
    ax[1, 1].hist(price.loc[idx], orientation="horizontal", alpha=0.5)

    ax[0, 1].set_visible(False)
    ax[0, 0].xaxis.set_visible(False)
    ax[1, 1].yaxis.set_visible(False)

    # ax[0,0].set_ylabel("Hist")
    # ax[1,1].set_xlabel("Hist")
    ax[1, 0].set_xlabel("Power in p.u.")
    ax[1, 0].set_ylabel("Price in €/MWh")

    fig.tight_layout(h_pad=0.1, w_pad=0.5)

    fig.legend(loc="center", bbox_to_anchor=(0.87, 0.85))

    return fig
