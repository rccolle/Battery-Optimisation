import pyomo.environ as pyo

class Battery(pyo.ConcreteModel):
    def __init__(self,
                 market_period,
                 max_power = 1,
                 max_stored_energy = 2,
                 round_trip_efficiency = 0.85,
                 cycle_cost = 0,
                 daily_cycle_limit = 1, 
                 power = 0, 
                 initial_stored_energy = 0, 
                 *args, 
                 **kwargs):
        
        pyo.ConcreteModel.__init__(self, *args, **kwargs)
        self.max_power = max_power                          # Maximum charge/discharge power (MW AC)
        self.max_stored_energy = max_stored_energy          # Maximum energy storage capacity (MWh DC)
        self.round_trip_efficiency = round_trip_efficiency  # Round trip efficiency (%)
        self.cycle_cost = cycle_cost                        # Energy cost per cycle ($/cycle)
        self.daily_cycle_limit = daily_cycle_limit          # Maximum cycles allowed per day (cycles/day)

        self.market_period = pyo.Set(initialize = market_period, ordered = True)    # Market period (datetime)
        
        # Battery variables
        self.stored_energy = pyo.Var(self.market_period, bounds = (0, self.max_stored_energy))   # Stored energy (MWh DC)
        self.power = pyo.Var(self.market_period, bounds = (-self.max_power, self.max_power))     # Charge/discharge power (MW AC)