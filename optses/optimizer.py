import numpy as np
import pandas as pd
import pyomo.environ as opt
import pyomo.dae as dae
from scipy.interpolate import interp1d
from pyomo.core.base.param import ScalarParam, IndexedParam


def build_lookup(profile):
    time = (profile.index - profile.index[0]).total_seconds() / 3600
    return interp1d(time, profile.values, kind="previous")


class OptModel:
    def __init__(self, solver, storage_model, profile, max_period_fec=1.5) -> None:
        self.model = opt.ConcreteModel()
        self.solver = solver
        self.storage_model = storage_model
        horizon = (profile.index[-1] - profile.index[0]).total_seconds() / 3600
        self.build_model(horizon, max_period_fec)
        self.discretize_model(profile)

    def build_model(self, horizon, max_period_fec):
        model = self.model

        # time parameters
        model.time = dae.ContinuousSet(bounds=(0, horizon))

        # price time series
        def profiles(block):
            model = block.model()
            block.price = opt.Param(
                model.time,
                within=opt.Reals,
                mutable=True,
                default=0,  # empty -> to be filled after discretization
            )

        # create sub-components
        model.profile = opt.Block(rule=profiles)
        model.bess = opt.Block(rule=self.storage_model.build)

        # add period throughput constraint
        model.max_period_fec = opt.Param(within=opt.NonNegativeReals, initialize=max_period_fec, mutable=True)

        @model.Integral(model.time)
        def fec(m, t):
            return (m.bess.powerc[t] + m.bess.powerd[t]) / 2 / m.bess.energy_capacity

        @model.Constraint()
        def throughput_constraint(m):
            return m.fec <= m.max_period_fec

        @model.Integral(model.time)
        def reveneue(m, t):
            return m.bess.power[t] * m.profile.price[t]

        @model.Objective()
        def cost(m):
            return m.reveneue

    def discretize_model(self, profile):
        nsteps = len(profile) - 1

        model = self.model
        discretizer = opt.TransformationFactory("dae.finite_difference")
        discretizer.apply_to(model, nfe=nsteps, scheme="FORWARD")

        # set initial price timeseries
        price = build_lookup(profile)
        for t in model.time:
            model.profile.price[t].set_value(float(price(t)))

    def recover_results(self):
        model = self.model
        return pd.DataFrame(
            data={
                "power": np.array([opt.value(model.bess.power[t]) for t in model.time]),
                "soc": np.array([opt.value(model.bess.soc[t]) for t in model.time]),
            },
            index=[t for t in model.time],
        )

    def update_model(self, val_dict: dict) -> None:
        model = self.model

        for block_name, block_values in val_dict.items():
            block = model.find_component(block_name)

            for param_name, val in block_values.items():
                param = block.find_component(param_name)
                if isinstance(param, ScalarParam):
                    param.set_value(val)
                elif isinstance(param, IndexedParam):
                    lut = build_lookup(val)  # create look-up table
                    for t in model.time:
                        param[t].set_value(float(lut(t)))

    def solve(self, *updates):
        if updates:
            for update in updates:
                self.update_model(update)

        try:
            solution = self.solver.solve(self.model)
            status = solution.solver.termination_condition
        except ValueError:
            status = "error"  # should not be necessary, but pyomo errors anyway

        return status
