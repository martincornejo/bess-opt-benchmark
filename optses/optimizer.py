import numpy as np
import pandas as pd
import pyomo.environ as opt
from pyomo.core.base.param import ScalarParam, IndexedParam


class OptModel:
    def __init__(self, solver, storage_model, profile, max_period_fec=1.0) -> None:
        self.model = opt.ConcreteModel()
        # self.solver = opt.SolverFactory("appsi_highs")
        self.solver = solver
        self.storage_model = storage_model
        self.build_model(profile, max_period_fec)

    def build_model(self, profile, max_period_fec):
        model = self.model

        # time parameters
        dt = (profile.index[1] - profile.index[0]).seconds / 3600
        timesteps = len(profile)
        model.time = opt.RangeSet(0, timesteps - 1)
        model.dt = opt.Param(initialize=dt)

        # price time series
        def profiles(block):
            model = block.model()
            block.price = opt.Param(
                model.time,
                within=opt.Reals,
                mutable=True,
                initialize=lambda b, t: profile.iloc[t],
            )

        # create sub-components
        model.profile = opt.Block(rule=profiles)
        model.bess = opt.Block(rule=self.storage_model.build)

        # add period throughput constraint
        model.max_period_fec = opt.Param(within=opt.NonNegativeReals, initialize=max_period_fec, mutable=True)

        @model.Expression()
        def fec(m):
            return sum(m.bess.powerc[t] + m.bess.powerd[t] for t in model.time) / 2 / m.bess.energy_capacity

        @model.Constraint()
        def throughput_constraint(m):
            return m.fec <= m.max_period_fec

        @model.Objective()
        def cost(m):
            return sum(m.bess.power[t] * m.profile.price[t] * model.dt for t in model.time)

    def recover_results(self):
        model = self.model
        return pd.DataFrame(
            data={
                "power": np.array([opt.value(model.bess.power[t]) for t in model.time]),
                "soc": np.array([opt.value(model.bess.soc[t]) for t in model.time]),
            }
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
                    for t in model.time:
                        param[t].set_value(val.iloc[t])

    def solve(self, *updates):
        if updates:
            for update in updates:
                self.update_model(update)

        status = self.solver.solve(self.model)
        return status["Solver"][0]["Termination condition"]

        # if status["Solver"][0]["Termination condition"] == "optimal":
        #     return self.recover_results()
        # else:
        #     return None  # TODO: best way to handle?
