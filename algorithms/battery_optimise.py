import pandas as pd
import numpy as np

import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

import pyomo.environ as pyo

def battery_optimisation(
        datetime,
        spot_price,
        initial_capacity = 0, 
        initial_cycles = 0,
        cycle_cost = 0,
        daily_cycle_limit = 1.0,
        include_revenue=True, 
        solver: str='appsi_highs'):
    """
    Determine optimal battery dispatch each five-minute period.
    
    ----------
    Parameters
    ----------
    datetime        : a list of time stamp
    spot_price      : a list of spot price of the corresponding time stamp
    initial_capacity : the initial capacity of the battery
    include_revenue : a boolean indicating if return results should include revenue calculation
    solver          : the name of the desire linear programming solver (e.g. 'glpk', 'mosek', 'gurobi')

    Returns
    ----------
    A dataframe that contains battery's opening capacity, spot price, and raw power for each period

    """
    # Battery's technical specification
    MIN_BATTERY_CAPACITY = 0.0
    MAX_BATTERY_CAPACITY = 10.0
    MAX_RAW_POWER = 5.0
    INITIAL_CAPACITY = initial_capacity # Default initial capacity will assume to be 0
    EFFICIENCY = 0.9
    MLF = 0.991 # Marginal Loss Factor
    HORIZON_CYCLE_LIMIT = daily_cycle_limit * ((datetime.iloc[-1] - datetime.iloc[0]).days + 1)

    df = pd.DataFrame({'datetime': datetime, 'spot_price': spot_price}).reset_index(drop=True)
    df['period'] = df.index
    
    # Define model and solver
    battery = pyo.ConcreteModel()
    opt = pyo.SolverFactory(solver)

    # defining components of the objective model
    # battery parameters
    battery.Period = pyo.Set(initialize=list(df.period), ordered=True)
    battery.Price = pyo.Param(initialize=list(df.spot_price), within=pyo.Any)

    # battery variables
    battery.Capacity = pyo.Var(battery.Period, bounds=(MIN_BATTERY_CAPACITY, MAX_BATTERY_CAPACITY))
    battery.Charge_power = pyo.Var(battery.Period, bounds=(0, MAX_RAW_POWER))
    battery.Discharge_power = pyo.Var(battery.Period, bounds=(0, MAX_RAW_POWER))
    battery.Cycles = pyo.Var(battery.Period, bounds=(0, HORIZON_CYCLE_LIMIT))
    battery.Cycle_cost = pyo.Var(battery.Period)

    # Set constraints for the battery
    # Defining the battery objective (function to be maximise)
    def maximise_profit(battery):
        rev = sum(df.spot_price[i] * (battery.Discharge_power[i] / 12) * MLF for i in battery.Period)
        cost = sum(df.spot_price[i] * (battery.Charge_power[i] / 12) / MLF for i in battery.Period)
        cost += sum(battery.Cycle_cost[i] for i in battery.Period)
        return rev - cost

    # Make sure the battery does not charge above the limit
    def over_charge(battery, i):
        return battery.Charge_power[i] <= (MAX_BATTERY_CAPACITY - battery.Capacity[i]) * 12 / EFFICIENCY

    # Make sure the battery discharge the amount it actually has
    def over_discharge(battery, i):
        return battery.Discharge_power[i] <= battery.Capacity[i] * 12 * EFFICIENCY

    # Make sure the battery do not discharge when price are not positive
    def negative_discharge(battery, i):
        # if the spot price is not positive, suppress discharge
        if battery.Price.extract_values_sparse()[None][i] <= 0:
            return battery.Discharge_power[i] == 0

        # otherwise skip the current constraint    
        return pyo.Constraint.Skip

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
            return battery.Cycles[i] == initial_cycles
        return battery.Cycles[i] == (battery.Cycles[i-1]
                                    + ((battery.Charge_power[i] / 12 * EFFICIENCY)
                                       + (battery.Discharge_power[i] / 12 / EFFICIENCY)) / 2 / MAX_BATTERY_CAPACITY)
    
    # Define cycle cost
    def incurred_cycle_cost(battery, i):
        if i == battery.Period.first():
            return battery.Cycle_cost[i] == battery.Cycles[i] * cycle_cost
        return battery.Cycle_cost[i] == cycle_cost * (battery.Cycles[i] - battery.Cycles[i-1])
        

    # Set constraint and objective for the battery
    battery.capacity_constraint = pyo.Constraint(battery.Period, rule=capacity_constraint)
    battery.over_charge = pyo.Constraint(battery.Period, rule=over_charge)
    battery.over_discharge = pyo.Constraint(battery.Period, rule=over_discharge)
    battery.negative_discharge = pyo.Constraint(battery.Period, rule=negative_discharge)
    battery.cycling_constraint = pyo.Constraint(battery.Period, rule=cycling_constraint)
    battery.cycle_cost_incurred = pyo.Constraint(battery.Period, rule=incurred_cycle_cost)

    battery.objective = pyo.Objective(rule=maximise_profit, sense=pyo.maximize)

    # Maximise the objective
    opt.solve(battery, tee=False)

    # unpack results
    charge_power, discharge_power, capacity, cycles, spot_price = ([] for i in range(5))
    for i in battery.Period:
        charge_power.append(battery.Charge_power[i].value)
        discharge_power.append(battery.Discharge_power[i].value)
        capacity.append(battery.Capacity[i].value)
        cycles.append(battery.Cycles[i].value)
        spot_price.append(battery.Price.extract_values_sparse()[None][i])

    result = pd.DataFrame(index=datetime,
                          data = {'spot_price':spot_price, 'charge_power':charge_power,
                                  'discharge_power':discharge_power, 'opening_capacity':capacity,
                                  'cycles': cycles})
    
    # make sure it does not discharge & charge at the same time
    if not len(result[(result.charge_power != 0) & (result.discharge_power != 0)]) == 0:
        print('Oops! The battery discharges & charges concurrently, the result has been returned')
        return result
    
    # convert columns charge_power & discharge_power to power
    result['power'] = np.where((result.charge_power > 0), 
                                -result.charge_power, 
                                result.discharge_power)
    

    # Determine closing capacity
    result['closing_capacity'] = (result['opening_capacity'] 
                                        + (result['charge_power'] / 12 * EFFICIENCY) 
                                        - (result['discharge_power'] / 12 / EFFICIENCY))

    
    result = result[['spot_price', 'power', 'opening_capacity', 'closing_capacity', 'cycles']]
    
    # calculate revenue
    if include_revenue:
        result['revenue'] = np.where(result.power < 0,
                                     result.power * result.spot_price / MLF,
                                     result.power * result.spot_price * MLF) / 12
    
    return result