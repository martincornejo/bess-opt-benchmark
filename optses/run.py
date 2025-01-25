# %%
import os
import sys
import multiprocessing

from datetime import datetime, timedelta
from configparser import ConfigParser

import numpy as np
import pandas as pd
import pyomo.environ as opt

from simses.main import SimSES
from tqdm import tqdm

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
def simses_factory(config, soh_r=1.0):
    # update params
    config["BATTERY"]["START_RESISTANCE_INC"] = f"{soh_r - 1}"

    # create simulation instance
    path = os.path.abspath(".")
    result_path = os.path.join(path, "results").replace("\\", "/") + "/"
    return SimSES(path=result_path, name="", simulation_config=config)


# %% [markdown]
# ## Optimizer


# %%
def build_linear_optimizer(profile):
    solver = opt.SolverFactory("appsi_highs")
    bess = LinearStorageModel(capacity=66e3, power=100e3, effc=0.9)
    return OptModel(solver=solver, storage_model=bess, profile=profile, max_period_fec=2)  # horizon = 24 h


# %%
def build_non_linear_optimizer(profile, soh_r=1.0):
    circuit = (217, 1)
    converter_params = {"k0": 0.00601144, "k1": 0.00863612, "k2": 0.01195589, "m0": 30}

    nl_storage = NonLinearStorageModel(
        battery_model=RintModel(
            capacity=94,
            r0=0.75e-3,
            soh_r=soh_r,
            v_bounds=(3.2, 4.2),
            i_bounds=(2 * 94, 2 * 94),
            circuit=circuit,
        ),
        converter_model=QuadraticLossConverter(power=100e3, **converter_params),
    )

    solver = opt.SolverFactory("ipopt")
    return OptModel(solver=solver, storage_model=nl_storage, profile=profile, max_period_fec=2)


# %%
def optimizer_factory(model, profile, soh_r=1.0):
    if model == "NL":
        optimizer = build_non_linear_optimizer(profile, soh_r=soh_r)
    elif model == "LP":
        optimizer = build_linear_optimizer(profile)
    else:
        raise NotImplementedError(f"{model} not supported.")
    return optimizer


# %% [markdown]
# ## MPC


# %%
def run_mpc(name, config_file, profile_file, sim_params, opt_params, horizon_hours=24, position=0):
    # load config
    config = ConfigParser()
    config.read(config_file)  # SimSES and time configs

    # time params
    timestep_sec = float(config["GENERAL"]["time_step"])
    timestep_dt = timedelta(seconds=timestep_sec)

    start_dt = datetime.strptime(config["GENERAL"]["START"], "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(config["GENERAL"]["END"], "%Y-%m-%d %H:%M:%S")
    horizon = timedelta(hours=horizon_hours, seconds=-timestep_sec)

    ## Price timeseries
    profile = load_price_timeseries(profile_file)

    ## SimSES
    simses = simses_factory(config, **sim_params)

    ## Optimizer
    optimizer = optimizer_factory(profile=profile.loc[start_dt : (start_dt + horizon)], **opt_params)

    ## MPC
    # initialization
    soc_sim = float(config["BATTERY"]["START_SOC"])  # start soc
    df = pd.DataFrame()

    # MPC loop
    timesteps = pd.date_range(start=start_dt, end=end_dt, freq=timestep_dt)
    pbar = tqdm(timesteps, desc=name, position=position)
    for t in pbar:  # iterate 1 day
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
            power_opt = round(res["power"].iloc[0])
            soc_opt = res["soc"].iloc[0]
            err_count = 0
        else:
            # if optimizer fails, take the continuation of the result of the previous iteration
            err_count += 1
            power_opt = res["power"].iloc[err_count]
            soc_opt = res["soc"].iloc[err_count]

        # simses
        simses_time = t + timestep_dt
        simses.run_one_simulation_step(time=simses_time.timestamp(), power=power_opt)

        soc_sim = simses.state.soc
        power_sim = simses.state.ac_power_delivered
        converter_losses = simses.state.pe_losses
        battery_losses = simses.state.storage_power_loss

        # write results
        data = {
            "soc_opt": soc_opt,
            "soc_sim": soc_sim,
            "power_opt": power_opt,
            "power_sim": power_sim,
            "converter_losses": converter_losses,
            "battery_losses": battery_losses,
        }
        df = pd.concat([df, pd.DataFrame(index=[t], data=[data])])
        pbar.refresh()

    simses.close()
    return df


# %%
def run_scenario(scenario: dict, results: dict, position: int = 0) -> None:
    """
    scenario: dict
        name and parameters of simulation scenario
    position: int
        task id to
    """
    name, params = scenario
    # print(f"Started {name}")
    results[name] = run_mpc(name, position=position, **params)
    # print(f"Finished {name}")


def run_pool(scenarios: dict) -> dict:
    num_cores = os.cpu_count()

    manager = multiprocessing.Manager()
    results = manager.dict()  # shared dictionary to store results

    tqdm.set_lock(multiprocessing.RLock())
    with multiprocessing.Pool(processes=num_cores, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        tasks = [(scenario, results, position) for position, scenario in enumerate(scenarios.items())]
        pool.starmap(run_scenario, tasks)

    return dict(results)


# %%
if __name__ == "__main__":
    config = "data/simulation.local.ini"
    profile = "data/electricity_prices_germany_2019.csv"

    scenarios = {}

    # for model in ("LP", "NL"):
    # for model in ("LP"):
    model = "LP"

    for R in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0):
        scenarios[f"{model} {R=}"] = {
            "config_file": config,
            "profile_file": profile,
            "sim_params": {"soh_r": R},
            "opt_params": {"model": model, "soh_r": R},
        }

    res = run_pool(scenarios)
