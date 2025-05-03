import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

Esys = 180e3  # Wh
Psys = 180e3  # W

colors = plt.cm.tab20c
cm_blues = plt.get_cmap("Blues")
cm_oranges = plt.get_cmap("Oranges")


def plot_eff_rev(df_lp, df_nl):
    fig, ax = plt.subplots()
    max_rev = 1  # max(df_lp["rev"].max(), df_nl["rev"].max())
    ax.scatter(df_lp["rte"], df_lp["rev"] / max_rev, label="LP")
    ax.scatter(df_nl["rte"], df_nl["rev"] / max_rev, label="NL")
    ax.legend()
    return fig


def plot_rev_bar(ax, df_lp, df_nl) -> None:
    r_values = sorted(df_lp["r"].unique())
    x = np.arange(len(r_values))
    bar_width = 0.35

    scale = Esys / 1e6

    df_lp = df_lp.sort_values(by="r")
    lp_rev = df_lp["rev"].to_numpy() / scale

    df_nl = df_nl.sort_values(by="r")
    nl_rev = df_nl["rev"].to_numpy() / scale

    ax.bar(x - bar_width / 2, lp_rev, width=bar_width, label="LP", color=colors(1))
    ax.bar(x + bar_width / 2, nl_rev, width=bar_width, label="NL", color=colors(6))

    for i in range(len(r_values)):
        improvement = (nl_rev[i] - lp_rev[i]) / lp_rev[i]
        ax.annotate(
            f"{improvement:+.1%}",
            xy=(x[i] - bar_width / 2, lp_rev[i]),
            xytext=(x[i] + bar_width / 2, nl_rev[i]),
            arrowprops=dict(arrowstyle="<|-", connectionstyle=f"bar,angle=0,fraction={0.3 - i * 0.3}", color="black"),
            ha="center",
            va="bottom",
        )

    ax.set_ylim(51000, 63000)
    ax.set_xticks(x)
    ax.set_xticklabels(r_values)
    ax.set_xlabel("SOH-R")
    ax.set_ylabel("Revenue / € / MW")
    ax.legend(title="Model")


def plot_eff_bar(ax, df_lp, df_nl) -> None:
    r_values = sorted(df_lp["r"].unique())
    x = np.arange(len(r_values))
    bar_width = 0.35

    df_lp = df_lp.sort_values(by="r")
    lp_loss_batt = df_lp["loss_batt"].to_numpy()
    lp_loss_conv = df_lp["loss_conv"].to_numpy()
    lp_loss_total = lp_loss_batt + lp_loss_conv

    df_nl = df_nl.sort_values(by="r")
    nl_loss_batt = df_nl["loss_batt"].to_numpy()
    nl_loss_conv = df_nl["loss_conv"].to_numpy()
    nl_loss_total = nl_loss_batt + nl_loss_conv

    ax.bar(x - bar_width / 2, lp_loss_conv, width=bar_width, label="LP - Converter", color=colors(0))
    ax.bar(x - bar_width / 2, lp_loss_batt, bottom=lp_loss_conv, width=bar_width, label="LP - Battery", color=colors(1))
    ax.bar(x + bar_width / 2, nl_loss_conv, width=bar_width, label="NL - Converter", color=colors(4))
    ax.bar(x + bar_width / 2, nl_loss_batt, bottom=nl_loss_conv, width=bar_width, label="NL - Battery", color=colors(6))

    for i in range(len(r_values)):
        improvement = nl_loss_total[i] - lp_loss_total[i]
        ax.annotate(
            f"{improvement:+.1%}",
            xy=(x[i] - bar_width / 2, lp_loss_total[i]),
            xytext=(x[i] + bar_width / 2, nl_loss_total[i]),
            arrowprops=dict(arrowstyle="<|-", connectionstyle=f"bar,angle=0,fraction={0.5 + i * 0.1}", color="black"),
            ha="center",
            va="bottom",
        )

    # ax.set_ylim(0, 0.176)
    ax.set_yticks(np.arange(0, 0.2, 0.025))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 100:.1f}"))
    ax.set_xticks(x)
    ax.set_xticklabels(r_values)
    ax.set_xlabel("SOH-R")
    ax.set_ylabel("Realative losses / %")
    ax.legend(ncol=2, loc="lower center", framealpha=1.0)


def plot_imb_bar(ax, df_lp, df_nl) -> None:
    r_values = sorted(df_lp["r"].unique())
    x = np.arange(len(r_values))
    bar_width = 0.35

    df_lp = df_lp.sort_values(by="r")
    lp_imb_c = df_lp["imb_under"].to_numpy() * 1e-3
    lp_imb_d = -df_lp["imb_over"].to_numpy() * 1e-3
    lp_total = lp_imb_c + lp_imb_d

    df_nl = df_nl.sort_values(by="r")
    nl_imb_c = df_nl["imb_under"].to_numpy() * 1e-3
    nl_imb_d = -df_nl["imb_over"].to_numpy() * 1e-3
    nl_total = nl_imb_c + nl_imb_d

    ax.bar(x - bar_width / 2, lp_imb_c, width=bar_width, label="LP - charge", color=colors(0))
    ax.bar(x - bar_width / 2, lp_imb_d, bottom=lp_imb_c, width=bar_width, label="LP - discharge", color=colors(1))
    ax.bar(x + bar_width / 2, nl_imb_c, width=bar_width, label="NL - charge", color=colors(4))
    ax.bar(x + bar_width / 2, nl_imb_d, bottom=nl_imb_c, width=bar_width, label="NL - discharge", color=colors(6))

    for i in range(len(r_values)):
        ax.text(x[i] - bar_width / 2, lp_total[i] + 0.01, f"{lp_total[i]:.0f} kWh", ha="center", va="bottom")
        ax.text(x[i] + bar_width / 2 + 0.08, nl_total[i] + 0.01, f"{nl_total[i]:.0f} kWh", ha="center", va="bottom")

    ax.set_ylim(0, 13500)
    ax.set_xticks(x)
    ax.set_xticklabels(r_values)
    ax.set_xlabel("SOH-R")
    ax.set_ylabel("Imbalance energy / kWh")
    ax.legend()


def plot_benchmark(df_lp, df_nl):
    fig, ax = plt.subplots(nrows=3, figsize=(4.5, 3 * 3.5))
    plot_rev_bar(ax[0], df_lp, df_nl)
    plot_eff_bar(ax[1], df_lp, df_nl)
    plot_imb_bar(ax[2], df_lp, df_nl)
    ax[0].set_title("a)", fontweight="bold", loc="left")
    ax[1].set_title("b)", fontweight="bold", loc="left")
    ax[2].set_title("c)", fontweight="bold", loc="left")
    # fig.tight_layout()
    return fig


def plot_power_ecdf(res_lp, res_nl):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    
    r_values = (1.0, 2.0, 3.0)
    for (i, r) in enumerate(r_values):
        df_lp = res_lp[r] / Psys
        ax.ecdf(df_lp["power_sim"], color=colors(2-i), label=fr"LP - $SOH_R = {i+1}$")

    for (i, r) in enumerate(r_values):
        df_nl = res_nl[r] / Psys
        ax.ecdf(df_nl["power_sim"], color=colors(6-i), label=fr"NL - $SOH_R = {i+1}$")
    
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 100:.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 100:.0f}"))
    ax.set_xlabel("Power / %")
    ax.set_ylabel("Cumulative frequency / %")
    ax.legend(loc="upper left")

    return fig 



def plot_timeseries(df, **kwargs):
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 8))

    for id in ["soc_sim", "soc_opt"]:
        axs[0].plot(df.index, df[id], label=id)

    axs[0].set_title("State of Charge")
    axs[0].set_ylabel("SOC")
    axs[0].legend()

    for id in ["power_sim", "power_opt"]:
        axs[1].plot(df.index, df[id], label=id)

    axs[1].set_title("Power")
    axs[1].set_ylabel("Power")
    axs[1].set_xlabel("Time")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    return fig


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
