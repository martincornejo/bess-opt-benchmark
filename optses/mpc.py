# %%
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
from nonlinear_model import NonLinearStorageModel, RintModel, QuadraticLossConverter, ConstantEfficiencyConverter


def load_price_timeseries(file: str) -> pd.Series:
    """
    Reads a price timeseries data from a CSV file.
    
    Parameters
    ----------
    file: str
        Location of CSV file.

    Returns
    -------
    pandas.Series
        A timeseries with ID1 price data in €/MWh.
    """
    df = pd.read_csv(file)
    df.index = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M")
    return df["Intraday Continuous 15 minutes ID1-Price"]


def simses_factory(start_soc: float, soh_r: float = 1.0):
    """
    Create simulation model based on parameters.
    
    Parameters
    ----------
    start_soc: float
        Initial SOC of the storage system at the start of the simulation.
    soh_r: float = 1.0
        Internal resistance increase of the battery in p.u., by default the battery is at begining-of-life (soh_r = 1).
    """
    # create simulation instance
    initial_states = {"start_soh_R": soh_r, "start_soc": start_soc, "start_T": 273.15 + 25}
    bat = Battery(Samsung94AhNMC(), circuit=(260, 2), initial_states=initial_states)
    return SinamicsS120(max_power=180e3, storage=bat)


def build_linear_optimizer(profile: pd.Series, max_fec: float = 2.0, eff: float = 0.95) -> OptModel:
    """
    Configures and creates a linear optimizer.
    
    Parameters
    ----------
    profile: pandas.Series
        Time series of price data. This is also used to define the optimization horizon and the time resolution of the steps.
    max_fec: float = 2.0
        Limit of full-equivalent-cycles (FEC) during the optimization horizon.
    eff: float = 0.95
        Unidirectional (charge or discharge) efficiency of the system in p.u., by default 95 %

    Returns
    -------
    OptModel
        An instance of the configured optimizer.
    """
    solver = opt.SolverFactory("appsi_highs")
    bess = LinearStorageModel(energy_capacity=180e3, power=180e3, effc=eff)
    return OptModel(solver=solver, storage_model=bess, profile=profile, max_period_fec=max_fec)


def build_non_linear_optimizer(profile: pd.Series, soh_r: float = 1.0, max_fec: float = 2.0) -> OptModel:
    """
    Configures and creates a non-linear optimizer.

    Parameters
    ----------
    profile: pandas.Series
        Time series of price data. This is also used to define the optimization horizon and the time resolution of the steps.
    soh_r: float = 1.0
        Internal resistance increase of the battery in p.u., by default the battery is at begining-of-life (soh_r = 1).
    max_fec: float = 2.0
        Limit of full-equivalent-cycles (FEC) during the optimization horizon.

    Returns
    -------
    OptModel
        An instance of the configured optimizer.
    """
    circuit = (260, 2)
    converter_params = {"k0": 0.00601144, "k1": 0.00863612, "k2": 0.01195589, "m0": 97}

    nl_storage = NonLinearStorageModel(
        energy_capacity=180e3,
        battery_model=RintModel(
            capacity=94,
            r0=0.75e-3,
            soh_r=soh_r,
            v_bounds=(2.7, 4.15),
            i_bounds=(2 * 94, 2 * 94),
            circuit=circuit,
        ),
        converter_model=QuadraticLossConverter(power=180e3, **converter_params),
        # converter_model=ConstantEfficiencyConverter(power=180e3, effc=0.97),
    )

    solver = opt.SolverFactory("bonmin")
    return OptModel(solver=solver, storage_model=nl_storage, profile=profile, max_period_fec=max_fec)


def optimizer_factory(model: str, profile: pd.Series, soh_r: float = 1.0, eff: float = 0.95, max_fec: float = 2.0) -> OptModel:
    """
    Configures and creates an optimizer based on the specified model type.

    Parameters
    ----------
    model : str
        The type of model to use for optimization. Supported models are:
        - "NL": Non-linear model.
        - "LP": Linear programming model.
    profile : pd.Series
        Time series of price data. This is used to define the optimization horizon and the time resolution of the steps.
    max_fec : float = 2.0
        Maximum feasible full equivalent cycles (FEC) during the optimization horizon, relevant for both models.
    soh_r : float = 1.0
        State of health ratio, relevant for the non-linear model.
    eff : float = 0.95
        Efficiency factor, relevant for the linear programming model.

    
    Returns
    -------
    OptModel
        An instance of the configured optimizer.
    """
    if model == "NL":
        optimizer = build_non_linear_optimizer(profile, soh_r=soh_r, max_fec=max_fec)
    elif model == "LP":
        optimizer = build_linear_optimizer(profile, eff=eff, max_fec=max_fec)
    else:
        raise NotImplementedError(f"{model} not supported.")
    return optimizer


def run_mpc(
    name: str,
    profile_file: str,
    sim_params: dict,
    opt_params: dict,
    horizon_hours: int = 12,
    timestep_sec: int = 900,
    total_time: timedelta | None = None,
    tqdm_options: dict | None = None,
) -> pd.DataFrame:
    """
    Simulates a receding horizon MPC operation. Iteratively, every 15 min, the optimizer receives the system states and future prices to schedule the operation for the next horizon. 
    The first 15 min of the schedule are passed to the system simulator which performs the 'real' dispatch and updates its state. 

    Parameters
    ----------
    name: str
        Simulation ID to display in the progress tracker.
    profile_file: str
        File location of the electricity price time series.
    sim_params: dict
        Parameters of the system simulator, see `simses_factory`.
    opt_params: dict
        Parameters of the optimizer, see `optimizer_factory`.
    horizon_hours: int = 12
        Time horizon of one optimization schedule in hours, by default 12.
    timestep_sec: int = 900
        The time resolution of the optimizer schedule in seconds, by default 900 (15 min).
    total_time: timedelta, optional
        Total MPC simulation horizon, by default the full time window of the input price time series is used.
    tqdm_options: dict, optional
        Configuration for the `tqdm` progress bar, by default {"position": 0}.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame with the timeseries of the optimizer and simulation results.
    """
    # progress bar default config
    if tqdm_options is None:
        tqdm_options = {"position": 0}

    # time params
    timestep_dt = timedelta(seconds=timestep_sec)
    horizon = timedelta(hours=horizon_hours, seconds=-timestep_sec)

    ## Price timeseries
    profile = load_price_timeseries(profile_file)
    profile = profile.resample(timestep_dt).ffill()
    start_dt: datetime = profile.index[0]
    if total_time is not None:
        profile = profile.loc[start_dt : (start_dt + total_time)]
    end_dt: datetime = profile.index[-1]

    ## SimSES
    simses = simses_factory(**sim_params)
    sim_steps = int(timestep_sec / 60)  #
    soc_sim = float(simses.storage.state.soc)  # start soc

    ## Optimizer
    optimizer = optimizer_factory(profile=profile.loc[start_dt : (start_dt + horizon)], **opt_params)
    steps = int(900 / timestep_sec)  # we run an optimization every 15 min

    ## MPC
    # initialization
    df = pd.DataFrame()
    res = pd.DataFrame()
    err_count = 0

    # MPC loop
    timesteps = pd.date_range(start=start_dt, end=(end_dt - horizon), freq=(timestep_dt * steps))
    for t in tqdm(timesteps, desc=name, **tqdm_options):
        # optses - solve optimal schedule
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
            power_opt_array = np.round(res["power"].iloc[0:steps].to_numpy())
            soc_opt_array = res["soc"].iloc[0:steps].to_numpy()
            err_count = 0
        else:
            # if optimizer fails, take the continuation of the result of the previous iteration
            err_count += 1
            offset = steps * err_count
            power_opt_array = np.round(res["power"].iloc[offset : (offset + steps)].to_numpy())
            soc_opt_array = res["soc"].iloc[offset : (offset + steps)].to_numpy()

        # simses - simulate the next 15 min
        for opt_step in range(steps):
            time_opt = t + (opt_step * timestep_dt)
            power_opt = power_opt_array[opt_step]
            soc_opt = soc_opt_array[opt_step]

            for sim_step in range(sim_steps):  # 1 min steps
                time = time_opt + (sim_step * timedelta(seconds=60))
                simses.update(power_setpoint=power_opt, dt=60)

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
