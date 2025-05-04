import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import PathPatch
from matplotlib.path import Path

Esys = 180e3  # Wh
Psys = 180e3  # W

colors = plt.cm.tab20c


## benchmark
def plot_eff_rev(df_lp, df_nl):
    fig, ax = plt.subplots()
    max_rev = max(df_lp["rev"].max(), df_nl["rev"].max())

    # linear regression
    rte = np.concatenate((df_lp["rte"], df_nl["rte"]))
    rev = np.concatenate((df_lp["rev"], df_nl["rev"])) / max_rev
    slope, intercept = np.polyfit(rte, rev, 1)
    r = np.corrcoef(rte, rev)[0, 1]
    print(f"{slope=:.2f}")
    print(f"{r=:.3f}")

    ax.axline((0, intercept), slope=slope, color="gray", alpha=0.5, linestyle="--")

    ax.scatter(df_lp["rte"], df_lp["rev"] / max_rev, label="LP")
    ax.scatter(df_nl["rte"], df_nl["rev"] / max_rev, label="NL")

    ax.set_ylim(0.87, 1.01)
    ax.set_xlim(0.83, 0.93)
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
            arrowprops=dict(arrowstyle="<|-", connectionstyle=f"bar,angle=0,fraction={0.3 - i * 0.25}", color="black"),
            ha="center",
            va="bottom",
        )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    ax.set_ylim(51000, 63000)
    ax.set_xticks(x)
    ax.set_xticklabels(r_values)
    ax.set_xlabel("$SOH_R$")
    ax.set_ylabel("Revenue / € / MW")
    ax.legend(title="Model", fontsize="small", frameon=False)


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

    ax.bar(x - bar_width / 2, lp_loss_conv, width=bar_width, label="LP - converter", color=colors(0))
    ax.bar(x - bar_width / 2, lp_loss_batt, bottom=lp_loss_conv, width=bar_width, label="LP - battery", color=colors(1))
    ax.bar(x + bar_width / 2, nl_loss_conv, width=bar_width, label="NL - converter", color=colors(4))
    ax.bar(x + bar_width / 2, nl_loss_batt, bottom=nl_loss_conv, width=bar_width, label="NL - battery", color=colors(6))

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

    ax.set_ylim(0, 0.2)
    ax.set_yticks(np.arange(0, 0.2, 0.025))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 100:.1f}"))
    ax.set_xticks(x)
    ax.set_xticklabels(r_values)
    ax.set_xlabel("$SOH_R$")
    ax.set_ylabel("Realative losses / %")
    # ax.legend(ncol=2, loc="lower center", framealpha=1.0, fontsize="small")
    legend = ax.legend(ncol=2, loc="upper left", fontsize="small", columnspacing=1.0, frameon=False)
    legend.set_zorder(0)


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

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    ax.set_ylim(0, 14000)
    ax.set_xticks(x)
    ax.set_xticklabels(r_values)
    ax.set_xlabel("$SOH_R$")
    ax.set_ylabel("Imbalance energy / kWh")
    legend = ax.legend(fontsize="small", frameon=False)
    legend.set_zorder(2)


def plot_benchmark(df_lp, df_nl):
    fig, ax = plt.subplots(nrows=3, figsize=(4.5, 3 * 2.6))
    plot_rev_bar(ax[0], df_lp, df_nl)
    plot_eff_bar(ax[1], df_lp, df_nl)
    plot_imb_bar(ax[2], df_lp, df_nl)
    fig.tight_layout()
    ax[0].set_title("a)", fontweight="bold", loc="left")
    ax[1].set_title("b)", fontweight="bold", loc="left")
    ax[2].set_title("c)", fontweight="bold", loc="left")
    return fig


def plot_power_ecdf(res_lp, res_nl):
    fig, ax = plt.subplots(figsize=(4.5, 2.3))
    ax2 = fig.add_axes([0.61, 0.21, 0.25, 0.45])

    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    vertices = [(-0.04, 0.81), (-0.04, 1.0), (1.04, 1.0), (1.04, 0.81), (0, 0)]
    path = Path(vertices, codes)
    pathpatch = PathPatch(path, facecolor="none", edgecolor="gray", linestyle="--")
    ax.add_patch(pathpatch)

    r_values = (1.0, 2.0, 3.0)
    for i, r in enumerate(r_values):
        df_lp = res_lp[r] / Psys
        ax.ecdf(df_lp["power_sim"], color=colors(2 - i), label=rf"LP - $SOH_R = {float(i + 1)}$")
        ax2.ecdf(df_lp["power_sim"], color=colors(2 - i))

    for i, r in enumerate(r_values):
        df_nl = res_nl[r] / Psys
        ax.ecdf(df_nl["power_sim"], color=colors(6 - i), label=rf"NL - $SOH_R = {float(i + 1)}$")
        ax2.ecdf(df_nl["power_sim"], color=colors(6 - i))

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 100:.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 100:.0f}"))
    ax.set_xlabel("Power / %")
    ax.set_ylabel("Cumulative frequency / %")
    ax.legend(loc="upper left", fontsize="small", frameon=False)

    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 100:.0f}"))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 100:.0f}"))
    ax2.set_ylim(0.81, 1.0)
    ax2.set_xlim(-0.05, 1.05)

    return fig


## sensitivity analysis
def plot_sensitivity(df_lp, df_nl, df_lp0, df_nl0):
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(4.5, 4.5))

    r_values = sorted(df_lp["r"].unique())

    for i, r in enumerate(r_values):
        ## LP
        # basis scenario
        df_lp0_ = df_lp0[df_lp0.r == r]
        rev_lp0 = df_lp0_["rev"].iloc[0]
        imb_lp0 = df_lp0_["imb_under"] + df_lp0_["imb_over"].abs()

        # sensitivity
        df_lp_ = df_lp[df_lp.r == r]
        df_lp_ = df_lp_.sort_values(by="eff")
        eff = df_lp_["eff"]
        imb_total_lp = (df_lp_["imb_under"] + df_lp_["imb_over"].abs()) / imb_lp0 - 1
        rev_lp = df_lp_["rev"] / rev_lp0 - 1

        ## NL
        # basis scenario
        df_nl0_ = df_nl0[df_nl0.r == r]
        rev_nl0 = df_nl0_["rev"].iloc[0]
        imb_nl0 = df_nl0_["imb_under"] + df_nl0_["imb_over"].abs()

        # sensitivity
        df_nl_ = df_nl[df_nl.r == r]
        df_nl_ = df_nl_.sort_values(by="r_opt")
        r_opt = df_nl_["r_opt"] - 1
        imb_total_nl = (df_nl_["imb_under"] + df_nl_["imb_over"].abs()) / imb_nl0 - 1
        rev_nl = df_nl_["rev"] / rev_nl0 - 1

        ax[0, 0].plot(eff, rev_lp, marker="o", color=colors(2 - i), label=rf"LP - $SOH_R = {float(i + 1)}$")
        ax[0, 1].plot(r_opt, rev_nl, marker="o", color=colors(6 - i), label=rf"NL - $SOH_R = {float(i + 1)}$")
        ax[1, 0].plot(eff, imb_total_lp, marker="o", color=colors(2 - i))
        ax[1, 1].plot(r_opt, imb_total_nl, marker="o", color=colors(6 - i))

    for i in range(2):
        ax[0, i].set_ylabel(r"$\Delta$Revenue / %")
        ax[0, i].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 100:.1f}"))
        ax[0, i].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 100:.1f}"))

    for i in range(2):
        ax[1, i].set_ylabel(r"$\Delta E_{imb}$ / %")
        ax[1, i].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 100:.1f}"))
        ax[1, i].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 100:,.0f}"))

    ax[0, 0].set_xlabel(r"$\eta$ / %")
    ax[1, 0].set_xlabel(r"$\eta$ / %")
    ax[0, 1].set_xlabel(r"$\Delta SOH_R$ / %")
    ax[1, 1].set_xlabel(r"$\Delta SOH_R$ / %")

    ax[0, 0].set_ylim(-0.02, 0.005)
    ax[0, 1].set_ylim(-0.02, 0.005)

    # second column has y-axis pointing right
    ax[1, 1].yaxis.set_label_position("right")
    ax[1, 1].yaxis.tick_right()
    ax[0, 1].yaxis.set_label_position("right")
    ax[0, 1].yaxis.tick_right()

    fig.legend(loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()

    ax[0, 0].set_title("a)", fontweight="bold", loc="left")
    ax[0, 1].set_title("b)", fontweight="bold", loc="left")
    ax[1, 0].set_title("c)", fontweight="bold", loc="left")
    ax[1, 1].set_title("d)", fontweight="bold", loc="left")
    return fig


## timeseries
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
