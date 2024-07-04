import pandas as pd

import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

import pyomo.environ as pyo

def battery_optimisation(datetime, energy_price, fcas_lower_price, fcas_raise_price, initial_capacity=0, solver: str='appsi_highs'):
    """
    Determine the optimal charge and discharge behavior of a battery based 
    in Victoria. Assuming pure foresight of future spot prices over every 
    half-hour period to maximise the revenue.
    PS: Assuming no degradation to the battery over the timeline and battery cannot
        charge and discharge concurrently.
    ----------
    Parameters
    ----------
    datetime        : a list of time stamp
    energy_price      : a list of spot price of the corresponding time stamp
    initial_capacit : the initial capacity of the battery
    include_revenue : a boolean indicates if return results should include revenue calculation
    solver          : the name of the desire linear programming solver (eg. 'glpk', 'mosek', 'gurobi')

    Returns
    ----------
    A dataframe that contains battery's opening capacity for each half-hour period, spot price
    of each half-hour period and battery's raw power for each half-hour priod
    """
    # Battery's technical specification
    MIN_BATTERY_CAPACITY = 0
    MAX_BATTERY_CAPACITY = 11
    MAX_RAW_POWER = 5
    MAX_FCAS_RAISE = 4
    MAX_FCAS_LOWER = 4
    INITIAL_CAPACITY = initial_capacity # Default initial capacity will assume to be 0
    EFFICIENCY = 0.9
    MLF = 0.991 # Marginal Loss Factor
    DAILY_CYCLE_LIMIT = 1.2 # Expressed as equivalent daily limit
    HORIZON_CYCLE_LIMIT = DAILY_CYCLE_LIMIT * (datetime.iloc[-1] - datetime.iloc[0]).days

    df = pd.DataFrame({'datetime': datetime,
                       'energy_price': energy_price,
                       'fcas_lower_price': fcas_lower_price,
                       'fcas_raise_price': fcas_raise_price}).reset_index(drop=True)
    df['period'] = df.index
    
    # Define model and solver
    battery = ConcreteModel()
    opt = SolverFactory(solver)

    # defining components of the objective model
    # battery parameters
    battery.Period = Set(initialize=list(df.period), ordered=True)
    battery.Energy_Price = Param(initialize=list(df.energy_price), within=Any)
    battery.FCAS_Raise_Price = Param(initialize=list(df.fcas_raise_price), within=Any)
    battery.FCAS_Lower_Price = Param(initialize=list(df.fcas_lower_price), within=Any)

    # battery variables
    battery.Capacity = Var(battery.Period, bounds=(MIN_BATTERY_CAPACITY, MAX_BATTERY_CAPACITY))
    battery.Charge_power = Var(battery.Period, bounds=(0, MAX_RAW_POWER))
    battery.Discharge_power = Var(battery.Period, bounds=(0, MAX_RAW_POWER))
    battery.FCAS_Raise_enablement = Var(battery.Period, bounds=(0, MAX_FCAS_RAISE))
    battery.FCAS_Lower_enablement = Var(battery.Period, bounds=(0, MAX_FCAS_LOWER))
    battery.Cycles = Var(battery.Period, bounds=(0, HORIZON_CYCLE_LIMIT))

    # Set constraints for the battery
    # Defining the battery objective (function to be maximised)
    def maximise_profit(battery):
        energy_revenue = sum(df.energy_price[i] * (battery.Discharge_power[i] / 12) * MLF for i in battery.Period)
        energy_cost = sum(df.energy_price[i] * (battery.Charge_power[i] / 12) / MLF for i in battery.Period)
        fcas_lower_revenue = sum(df.fcas_lower_price * (battery.FCAS_Lower_enablement[i] / 12) * MLF for i in battery.Period)
        fcas_raise_revenue = sum(df.fcas_raise_price * (battery.FCAS_Raise_enablement[i] / 12) * MLF for i in battery.Period)
        return energy_revenue - energy_cost + fcas_lower_revenue + fcas_raise_revenue

    # Make sure the battery does not charge above the limit
    def over_charge(battery, i):
        return battery.Charge_power[i] <= (MAX_BATTERY_CAPACITY - battery.Capacity[i]) * 12 / EFFICIENCY

    # Make sure the battery only discharges the amount it actually has
    def over_discharge(battery, i):
        return battery.Discharge_power[i] <= battery.Capacity[i] * 12 * EFFICIENCY

    # Make sure the battery does not discharge when price is not positive
    def negative_discharge(battery, i):
        # if the spot price is not positive, suppress discharge
        if battery.Energy_Price.extract_values_sparse()[None][i] <= 0:
            return battery.Discharge_power[i] == 0

        # otherwise skip the current constraint    
        return Constraint.Skip

    # Defining capacity rule for the battery
    def capacity_constraint(battery, i):
        # Assigning battery's starting capacity at the beginning
        if i == battery.Period.first():
            return battery.Capacity[i] == INITIAL_CAPACITY
        # if not update the capacity normally    
        return battery.Capacity[i] == (battery.Capacity[i-1] 
                                        + (battery.Charge_power[i-1] / 12 * EFFICIENCY) 
                                        - (battery.Discharge_power[i-1] / 12 / EFFICIENCY))

    # Defining battery cycling limit
    def cycling_constraint(battery, i):
        if i == battery.Period.first():
            return battery.Cycles[i] == 0
        return battery.Cycles[i] == (battery.Cycles[i-1]
                                    + ((battery.Charge_power[i] / 12 * EFFICIENCY)
                                       + (battery.Discharge_power[i] / 12)) / 2 / MAX_BATTERY_CAPACITY)

    # Energy + FCAS enablement should not exceed battery capacity
    def discharge_plus_raise_limit(battery, i):
        return battery.FCAS_Raise_enablement[i] + battery.Discharge_power[i] <= MAX_RAW_POWER
    
    def charge_plus_lower_limit(battery, i):
        return battery.FCAS_Lower_enablement[i] + battery.Charge_power[i] <= MAX_RAW_POWER

    # Set constraint and objective for the battery
    battery.capacity_constraint = Constraint(battery.Period, rule=capacity_constraint)
    battery.over_charge = Constraint(battery.Period, rule=over_charge)
    battery.over_discharge = Constraint(battery.Period, rule=over_discharge)
    battery.negative_discharge = Constraint(battery.Period, rule=negative_discharge)
    battery.cycling_constraint = Constraint(battery.Period, rule=cycling_constraint)
    battery.discharge_plus_raise_limit = Constraint(battery.Period, rule=discharge_plus_raise_limit)
    battery.charge_plus_lower_limit = Constraint(battery.Period, rule=charge_plus_lower_limit)
    battery.objective = Objective(rule=maximise_profit, sense=maximize)

    # Maximise the objective
    opt.solve(battery, tee=False)

    # unpack results
    charge_power, discharge_power, capacity, cycles, energy_price = ([] for i in range(5))
    for i in battery.Period:
        charge_power.append(battery.Charge_power[i].value)
        discharge_power.append(battery.Discharge_power[i].value)
        capacity.append(battery.Capacity[i].value)
        cycles.append(battery.Cycles[i].value)
        energy_price.append(battery.Energy_Price.extract_values_sparse()[None][i])

    result = pd.DataFrame(index=datetime,
                          data = {'energy_price':energy_price, 'charge_power':charge_power,
                                  'discharge_power':discharge_power, 'opening_capacity':capacity,
                                  'cycles': cycles})
    
    # make sure it does not discharge & charge at the same time
    if not len(result[(result.charge_power != 0) & (result.discharge_power != 0)]) == 0:
        print('Ops! The battery discharges & charges concurrently, the result has been returned')
      
    return result