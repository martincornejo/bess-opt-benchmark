import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def calc_fec(res):
    return res["soc_sim"].diff().abs().sum() / 2


def calc_roundtrip_efficiency(res):
    p = res["power_sim"]
    e_pos = p[p > 0].sum() * 0.25  # Wh
    e_neg = p[p < 0].sum() * 0.25  # Wh

    delta_soc = res["soc_sim"].iloc[-1] - res["soc_sim"].iloc[0]
    delta_e = delta_soc * 66e3  # Wh

    return np.abs(e_neg) / (e_pos - delta_e)


def calc_revenue(res, price):
    df = res.join(price)
    return -1 * sum(df["power_sim"] * df["Intraday Continuous 15 minutes ID1-Price"]) * 0.25 * 1e-6  # W -> MWh


def summary_results(results, price):
    df = pd.DataFrame()
    for name, res in results.items():
        data = {"name": name, "FEC": calc_fec(res), "Eff": calc_roundtrip_efficiency(res), "Rev": calc_revenue(res, price)}
        df = pd.concat([df, pd.DataFrame(data=[data])])
    return df


def print_performance(name, res, price):
    print(f"-- {name} --")
    fec = calc_fec(res)
    eff = calc_roundtrip_efficiency(res)
    rev = calc_revenue(res, price)
    print(f"Full equivalent cycles: {fec:.1f}")
    print(f"Roundtrip efficiency: {eff:.2%}")
    print(f"Revenue: {rev:.2f}€")
    print("")


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
