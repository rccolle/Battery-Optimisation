import pyomo.environ as pyo

class Battery(pyo.ConcreteModel):
    def __init__(self, max_power = 1, max_stored_energy = 2, round_trip_efficiency = 0.85, cycle_cost = 0, daily_cycle_limit = 1, power = 0, stored_energy = 0):
        self.max_power = max_power                          # Maximum charge/discharge power (MW AC)
        self.max_stored_energy = max_stored_energy          # Maximum energy storage capacity (MWh DC)
        self.power = power                                  # Current charge/discharge power (MW AC)
        self.stored_energy = stored_energy                  # Current stored energy (MWh DC)
        self.round_trip_efficiency = round_trip_efficiency  # Round trip efficiency (%)
        self.cycle_cost = cycle_cost                        # Energy cost per cycle ($/cycle)
        self.daily_cycle_limit = daily_cycle_limit          # Maximum cycles allowed per day (cycles/day)