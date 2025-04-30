import pyomo.environ as opt


class NonLinearStorageModel:
    def __init__(self, battery_model, converter_model, energy_capacity: float) -> None:
        self.energy_capacity = energy_capacity
        self.battery_model = battery_model
        self.converter_model = converter_model

    def build(self, block) -> None:
        self.battery_model.build(block)
        self.converter_model.build(block)
        block.energy_capacity = opt.Param(within=opt.NonNegativeReals, initialize=self.energy_capacity, mutable=True)


class RintModel:
    "Internal resistance equivalent circuit model"

    def __init__(
        self,
        capacity: float,
        r0: float,
        circuit: tuple[int, int],
        v_bounds: tuple[float, float],
        i_bounds: tuple[float, float] | None = None,
        soc_bounds: tuple[float, float] = (0.0, 1.0),
        soc_start: float = 0.5,
        soh_r: float = 1.0,
        eff: float = 0.9999,
    ) -> None:
        # circuit
        self._circuit = circuit

        # params
        self._capacity = capacity
        self._r0 = r0
        self._v_bounds = v_bounds
        self._soc_bounds = soc_bounds
        self._soc_start = soc_start
        self._soh_r = soh_r
        self._eff = eff

        if i_bounds is None:
            i_bounds = (capacity, capacity)
        self._i_bounds = i_bounds

    def build(self, block) -> None:
        model = block.model()

        # params
        (serial, parallel) = self._circuit
        block.capacity = opt.Param(initialize=self._capacity * parallel, mutable=True)  # Ah * p

        (soc_min, soc_max) = self._soc_bounds
        block.soc_min = opt.Param(within=opt.NonNegativeReals, initialize=soc_min, mutable=True)
        block.soc_max = opt.Param(within=opt.NonNegativeReals, initialize=soc_max, mutable=True)
        block.soc_start = opt.Param(within=opt.NonNegativeReals, initialize=self._soc_start, mutable=True)

        block.effc = opt.Param(within=opt.PercentFraction, initialize=self._eff)
        block.effd = opt.Param(within=opt.PercentFraction, initialize=self._eff)

        block.r0 = opt.Param(initialize=self._r0 / parallel * serial)
        block.soh_r = opt.Param(initialize=self._soh_r, mutable=True)

        @block.Expression()
        def r(b):
            return b.r0 * b.soh_r

        # vars
        block.soc = opt.Var(model.time, within=opt.UnitInterval, bounds=(block.soc_min, block.soc_max))

        (imax_c, imax_d) = self._i_bounds
        block.ic = opt.Var(model.time, within=opt.NonNegativeReals, bounds=(0, imax_c * parallel))
        block.id = opt.Var(model.time, within=opt.NonNegativeReals, bounds=(0, imax_d * parallel))

        @block.Expression(model.time)
        def i(b, t):
            return b.ic[t] - b.id[t]

        def ocv_curve(soc):
            a1 = 3.3479
            a2 = -6.7241
            a3 = 2.5958
            a4 = -61.9684
            b1 = 0.6350
            b2 = 1.4376
            k0 = 4.5868
            k1 = 3.1768
            k2 = -3.8418
            k3 = -4.6932
            k4 = 0.3618
            k5 = 0.9949

            return (
                k0
                + k1 / (1 + opt.exp(a1 * (soc - b1)))
                + k2 / (1 + opt.exp(a2 * (soc - b2)))
                + k3 / (1 + opt.exp(a3 * (soc - 1)))
                + k4 / (1 + opt.exp(a4 * soc))
                + k5 * soc
            ) 

        @block.Expression(model.time)
        def ocv(b, t):
            # if t == model.time.first():
            #     soc = (b.soc[t] + b.soc_start) / 2
            #     # soc = b.soc_start
            # else:
            #     soc = (b.soc[t] + b.soc[t - 1]) / 2
            #     # soc = b.soc[t - 1]
            soc = b.soc[t]
            return ocv_curve(soc) * serial

        (v_min, v_max) = self._v_bounds
        block.v = opt.Var(model.time, within=opt.NonNegativeReals, bounds=(v_min * serial, v_max * serial))

        @block.Constraint(model.time)
        def v_constraint(b, t):
            return b.v[t] == b.ocv[t] + b.r * b.i[t]

        # constraints
        @block.Constraint(model.time)
        def soc_constraint(b, t):
            if t == model.time.first():
                return b.soc[t] == b.soc_start + model.dt * (b.ic[t] * b.effc - b.id[t] * (1 / b.effd)) / b.capacity
            return b.soc[t] == b.soc[t - 1] + model.dt * (b.ic[t] * b.effc - b.id[t] * (1 / b.effd)) / b.capacity

        @block.Expression(model.time)
        def power_dc(b, t):
            return b.v[t] * b.i[t]

        @block.Constraint()
        def power_end_constraint(b):
            t = model.time.last()
            return b.power_dc[t] == 0.0


class QuadraticLossConverter:
    def __init__(self, power, k0, k1, k2, m0) -> None:
        self._power = power
        self._k0 = k0  # * power
        self._k1 = k1
        self._k2 = k2  # / power
        self._m0 = m0  # / power

    def build(self, block) -> None:
        model = block.model()

        block.k0 = opt.Param(within=opt.Reals, initialize=self._k0)
        block.k1 = opt.Param(within=opt.Reals, initialize=self._k1)
        block.k2 = opt.Param(within=opt.Reals, initialize=self._k2)
        block.m0 = opt.Param(within=opt.Reals, initialize=self._m0)

        block.power_max = opt.Param(within=opt.NonNegativeReals, initialize=self._power)

        block.powerc = opt.Var(model.time, bounds=(0, block.power_max))
        block.powerd = opt.Var(model.time, bounds=(0, block.power_max))

        @block.Expression(model.time)
        def power(b, t):
            return b.powerc[t] - b.powerd[t]

        @block.Expression(model.time)
        def power_factor(b, t):
            return (b.powerc[t] + b.powerd[t]) / b.power_max

        @block.Expression(model.time)
        def converter_loss(b, t):
            return (
                b.k0 * (1 - opt.exp(-b.m0 * b.power_factor[t]))  # constant loss + activation
                + b.k1 * b.power_factor[t]
                + b.k2 * b.power_factor[t] ** 2
            ) * b.power_max

        @block.Constraint(model.time)
        def converter_loss_constraint(b, t):
            return b.power_dc[t] == b.power[t] - b.converter_loss[t]


class ConstantEfficiencyConverter:
    def __init__(self, power: float, effc: float, effd: float | None = None) -> None:
        if effd is None:
            effd = effc

        self._power = power
        self._effc = effc
        self._effd = effd

    def build(self, block):
        model = block.model()

        block.power_max = opt.Param(within=opt.NonNegativeReals, initialize=self._power)
        block.acdc_effc = opt.Param(within=opt.PercentFraction, initialize=self._effc, mutable=True)
        block.acdc_effd = opt.Param(within=opt.PercentFraction, initialize=self._effd, mutable=True)

        block.powerc = opt.Var(model.time, bounds=(0, block.power_max))
        block.powerd = opt.Var(model.time, bounds=(0, block.power_max))

        @block.Expression(model.time)
        def power(b, t):
            return b.powerc[t] - b.powerd[t]

        @block.Constraint(model.time)
        def conversion_losses_constraint(b, t):
            return b.power_dc[t] == b.powerc[t] * b.acdc_effc - b.powerd[t] / b.acdc_effd


# def recover_ecm(model):
#     return pd.DataFrame(
#         data={
#             "power": np.array([opt.value(model.bess.power[t]) for t in model.time]),
#             "power_dc": np.array([opt.value(model.bess.power_dc[t]) for t in model.time]),
#             "soc": np.array([opt.value(model.bess.soc[t]) for t in model.time]),
#             "i": np.array([opt.value(model.bess.i[t]) for t in model.time]),
#             "v": np.array([opt.value(model.bess.v[t]) for t in model.time]),
#             "ocv": np.array([opt.value(model.bess.ocv[t]) for t in model.time]),
#         }
#     )
