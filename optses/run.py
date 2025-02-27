# %%
import os
import sys
import multiprocessing

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyomo.environ as opt
from tqdm import tqdm

from simses.battery.battery import Battery
from simses.model.cell.samsung94Ah_nmc import Samsung94AhNMC
from simses.model.converter.sinamics import SinamicsS120

from optimizer import OptModel
from linear_model import LinearStorageModel
from nonlinear_model import NonLinearStorageModel, RintModel, QuadraticLossConverter

# %% [markdown]
# ## Price timeseries


# %%
def load_price_timeseries(file: str) -> pd.Series:
    df = pd.read_csv(file)
    df.index = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M")
    return df["Intraday Continuous 15 minutes ID1-Price"]  # * 1e-6 # €/MWh -> €/Wh


# %% [markdown]
# ## SimSES


# %%
def simses_factory(start_soc, soh_r=1.0):
    # create simulation instance
    initial_states = {"start_soh_R": soh_r, "start_soc": start_soc, "start_T": 273.15 + 25}
    bat = Battery(Samsung94AhNMC(), circuit=(260, 2), initial_states=initial_states)
    return SinamicsS120(max_power=180e3, storage=bat)


# %% [markdown]
# ## Optimizer


# %%
def build_linear_optimizer(profile, max_fec=2.0, eff=0.95):
    solver = opt.SolverFactory("appsi_highs")
    bess = LinearStorageModel(energy_capacity=180e3, power=180e3, effc=eff)
    return OptModel(solver=solver, storage_model=bess, profile=profile, max_period_fec=max_fec)


# %%
def build_non_linear_optimizer(profile, soh_r=1.0, max_fec=2.0):
    circuit = (260, 2)
    converter_params = {"k0": 0.00601144, "k1": 0.00863612, "k2": 0.01195589, "m0": 97}

    nl_storage = NonLinearStorageModel(
        energy_capacity=180e3,
        battery_model=RintModel(
            capacity=94,
            r0=0.75e-3,
            soh_r=soh_r,
            v_bounds=(3.2, 4.2),
            i_bounds=(2 * 94, 2 * 94),
            circuit=circuit,
        ),
        converter_model=QuadraticLossConverter(power=180e3, **converter_params),
    )

    solver = opt.SolverFactory("bonmin")
    return OptModel(solver=solver, storage_model=nl_storage, profile=profile, max_period_fec=max_fec)


# %%
def optimizer_factory(model, profile, soh_r=1.0, eff=0.95, max_fec=2.0):
    if model == "NL":
        optimizer = build_non_linear_optimizer(profile, soh_r=soh_r, max_fec=max_fec)
    elif model == "LP":
        optimizer = build_linear_optimizer(profile, eff=eff, max_fec=max_fec)
    else:
        raise NotImplementedError(f"{model} not supported.")
    return optimizer


# %% [markdown]
# ## MPC


# %%
def run_mpc(name, profile_file, sim_params, opt_params, horizon_hours=12, steps=1, position=0):
    # time params
    timestep_sec = 900
    timestep_dt = timedelta(seconds=timestep_sec)
    horizon = timedelta(hours=horizon_hours, seconds=-timestep_sec)

    ## Price timeseries
    profile = load_price_timeseries(profile_file)
    # profile = profile.resample("5Min").ffill()
    start_dt: datetime = profile.index[0]
    profile = profile.loc[start_dt : (start_dt + timedelta(days=1))]
    end_dt: datetime = profile.index[-1]

    ## SimSES
    simses = simses_factory(**sim_params)
    soc_sim = float(simses.storage.state.soc)  # start soc

    ## Optimizer
    optimizer = optimizer_factory(profile=profile.loc[start_dt : (start_dt + horizon)], **opt_params)

    ## MPC
    # initialization
    df = pd.DataFrame()
    err_count = 0

    # MPC loop
    timesteps = pd.date_range(start=start_dt, end=(end_dt - horizon), freq=(timestep_dt * steps))
    for t in tqdm(timesteps, desc=name, position=position, mininterval=1):
        # optses
        timerange = pd.date_range(start=t, end=t + horizon, freq=timestep_dt)
        params = {
            "bess": {
                "soc_start": soc_sim,
            },
            "profile": {
                "price": profile.loc[timerange],
            },
        }

        status = optimizer.solve(params)
        if status == "optimal":
            res = optimizer.recover_results()
            power_opt_array = np.round(res["power"].iloc[0:steps])
            soc_opt_array = res["soc"].iloc[0:steps]
            err_count = 0
        else:
            # if optimizer fails, take the continuation of the result of the previous iteration
            err_count += 1
            power_opt_array = res["power"].iloc[(steps * err_count) : (steps * err_count + steps)]
            soc_opt_array = res["soc"].iloc[(steps * err_count) : (steps * err_count + steps)]

        # simses
        for step in range(steps):
            time = t + (step * timestep_dt)
            power_opt = power_opt_array[step]
            soc_opt = soc_opt_array[step]

            simses.update(power_setpoint=power_opt, dt=timestep_sec)

            soc_sim = simses.storage.state.soc
            power_sim = simses.state.power
            converter_losses = simses.state.loss
            battery_losses = simses.storage.state.loss

            # write results
            data = {
                "soc_opt": soc_opt,
                "soc_sim": soc_sim,
                "power_opt": power_opt,
                "power_sim": power_sim,
                "converter_losses": converter_losses,
                "battery_losses": battery_losses,
            }
            df = pd.concat([df, pd.DataFrame(index=[time], data=[data])])

    return df


# %%
def run_scenario(scenario: dict, position: int = 0) -> None:
    """
    scenario: dict
        name and parameters of simulation scenario
    position: int
        task id to
    """
    name, params = scenario
    df = run_mpc(name, position=position, **params)
    df.index.name = "time"
    df.to_parquet(f"results/{name}.parquet")


def run_parallel(scenarios: dict) -> None:
    num_cores = 4  # int(os.cpu_count() / 2)

    tqdm.set_lock(multiprocessing.RLock())
    with multiprocessing.Pool(processes=num_cores, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        tasks = [(scenario, position) for position, scenario in enumerate(scenarios.items())]
        pool.starmap(run_scenario, tasks)


# %%
def main():
    scenarios = {}

    year = 2021
    fec = 2.0

    model = "LP"
    for r in (1.0, 1.5, 2.0, 3.0):
        for eff in (0.93, 0.94, 0.95, 0.96):
            scenarios[f"{year} {model} {r=} {eff=}"] = {
                "profile_file": f"data/intraday_prices/electricity_prices_germany_{year}.csv",
                "sim_params": {"start_soc": 0.0, "soh_r": r},
                "opt_params": {"model": model, "eff": eff, "max_fec": fec},
            }

    model = "NL"
    for r_sim in (1.0, 1.5, 2.0, 3.0):
        for r_opt in (1.0, 1.5, 2.0, 3.0):
            scenarios[f"{year} {model} {r=} {eff=}"] = {
                "profile_file": f"data/intraday_prices/electricity_prices_germany_{year}.csv",
                "sim_params": {"start_soc": 0.0, "soh_r": r_sim},
                "opt_params": {"model": model, "soh_r": r_opt, "max_fec": fec},
            }

    run_parallel(scenarios)


if __name__ == "__main__":
    main()
