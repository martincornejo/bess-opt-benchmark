import pyomo.environ as opt
import pyomo.dae as dae


class LinearStorageModel:
    """Class to construct storage model"""

    def __init__(
        self,
        power: float,
        energy_capacity: float,
        soc_start: float = 0.5,
        soc_bounds: tuple[float, float] = (0.0, 1.0),
        effc: float = 0.9,
        effd: float | None = None,
    ) -> None:
        if effd is None:
            effd = effc

        self._energy_capacity = energy_capacity  # Wh
        self._power = power  # W
        self._soc_start = soc_start
        self._soc_bounds = soc_bounds
        self._effc = effc  # charge efficiency
        self._effd = effd  # discharge efficiency

    def build(self, block) -> None:
        model = block.model()

        ## Params
        block.max_power = opt.Param(within=opt.NonNegativeReals, initialize=self._power, mutable=True)
        block.energy_capacity = opt.Param(within=opt.NonNegativeReals, initialize=self._energy_capacity, mutable=True)
        block.soc_min = opt.Param(within=opt.NonNegativeReals, initialize=self._soc_bounds[0], mutable=True)
        block.soc_max = opt.Param(within=opt.NonNegativeReals, initialize=self._soc_bounds[1], mutable=True)
        block.soc_start = opt.Param(within=opt.NonNegativeReals, initialize=self._soc_start, mutable=True)
        block.effc = opt.Param(within=opt.PercentFraction, initialize=self._effc, mutable=True)
        block.effd = opt.Param(within=opt.PercentFraction, initialize=self._effd, mutable=True)

        ## Variables + Constraints
        block.soc = opt.Var(model.time, bounds=(block.soc_min, block.soc_max))
        block.dsoc = dae.DerivativeVar(sVar=block.soc, wrt=model.time)

        block.powerc = opt.Var(model.time, bounds=(0, block.max_power))
        block.powerd = opt.Var(model.time, bounds=(0, block.max_power))

        @block.Expression(model.time)
        def power(b, t):
            return b.powerc[t] - b.powerd[t]

        @block.Constraint(model.time)
        def soc_constraint(b, t):
            return b.dsoc[t] == (b.powerc[t] * b.effc - b.powerd[t] / b.effd) / b.energy_capacity

        @block.Constraint()
        def initial_soc(b):
            t = model.time.first()
            return b.soc[t] == b.soc_start
