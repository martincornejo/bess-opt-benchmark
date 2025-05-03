import re
import pandas as pd

def extract_value(string: str, key: str) -> float:
    "Extract parameter from simulation ID"
    match = re.search(fr"{key}=([\d.]+)", string)
    if match:
        value = match.group(1)
    return float(value)

def calc_fec(df: pd.DataFrame) -> float:
    "Full equivalent cycles (FEC) perfomed by the storage system"
    power = df["power_sim"]
    power_pos = power[power > 0].sum() * (1 / 60)
    power_neg = power[power < 0].abs().sum() * (1 / 60)
    return (power_pos + power_neg) / 2 / 180e3

def calc_rte(df: pd.DataFrame) -> float:
    "Roundtrip efficiency"
    p = df["power_sim"]
    e_pos = p[p > 0].abs().sum() * (1 / 60)  # Wh
    e_neg = p[p < 0].abs().sum() * (1 / 60)  # Wh

    delta_soc = df["soc_sim"].iloc[-1] - df["soc_sim"].iloc[0]
    delta_e = delta_soc * 180e3  # Wh

    return abs(e_neg) / (e_pos - delta_e)

def calc_revenue(df: pd.DataFrame, price: pd.Series) -> float:
    "Total revenue from trading in €"
    price = price.resample("1Min").ffill()
    df = df.join(price)
    return -1 * sum(df["power_sim"] * df["Intraday Continuous 15 minutes ID1-Price"]) * (1/60) * 1e-6  # W -> MWh

def calc_imbalance_pos(df: pd.DataFrame) -> float:
    df["imb"] = -(df["power_opt"] - df["power_sim"]) * (1 / 60) #* 1e-6 # MWh
    # negation since positive power is charging
    return df.loc[df.imb > 0, "imb"].sum() # BESS under-supply

def calc_imbalance_neg(df: pd.DataFrame) -> float:
    df["imb"] = -(df["power_opt"] - df["power_sim"]) * (1 / 60) #* 1e-6 # MWh
    # negation since positive power is charging
    return df.loc[df.imb < 0, "imb"].sum() # BESS over-supply

def calc_loss_batt(df: pd.DataFrame) -> float:
    power_out = df[df["power_sim"] < 0]["power_sim"].abs().sum()
    loss_batt = df["battery_losses"].sum()
    loss_conv = df["converter_losses"].sum()
    loss_total = loss_batt + loss_conv
    return loss_batt / (power_out + loss_total)

def calc_loss_conv(df: pd.DataFrame) -> float:
    power_out = df[df["power_sim"] < 0]["power_sim"].abs().sum()
    loss_batt = df["battery_losses"].sum()
    loss_conv = df["converter_losses"].sum()
    loss_total = loss_batt + loss_conv
    return loss_conv / (power_out + loss_total)

def analyze_results_lp(res: dict[str, pd.DataFrame], price: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame()
    for (id, df) in res.items():
        if "LP" in id:
            data = dict(
                r = extract_value(id, "r"),
                eff = extract_value(id, "eff"),
                rev = calc_revenue(df, price),
                fec = calc_fec(df),
                rte = calc_rte(df),
                loss_batt = calc_loss_batt(df),
                loss_conv = calc_loss_conv(df),
                imb_under = calc_imbalance_pos(df),
                imb_over = calc_imbalance_neg(df),
            )
            out = pd.concat([out, pd.DataFrame(data=[data])])

    return out


def analyze_results_nl(res: dict[str, pd.DataFrame], price: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame()
    for (id, df) in res.items():
        if ("NL" in id):
            data = dict(
                r = extract_value(id, "r"),
                r_opt = extract_value(id, "r_opt"),
                rev = calc_revenue(df, price),
                fec = calc_fec(df),
                rte = calc_rte(df),
                loss_batt = calc_loss_batt(df),
                loss_conv = calc_loss_conv(df),
                imb_under = calc_imbalance_pos(df),
                imb_over = calc_imbalance_neg(df),
            )
            out = pd.concat([out, pd.DataFrame(data=[data])])

    return out