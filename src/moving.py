from tqdm import tqdm
import os
import numpy as np 
import pandas as pd 
import uuid
import json
import sys
import pickle

import warnings 
warnings.filterwarnings("ignore")

import gurobipy as gp
from gurobipy import GRB
from baseline.utils import load_problem_data

from baseline.evaluation_v6 import (
        get_capacity_by_server_generation_latency_sensitivity,
        check_datacenter_slots_size_constraint,
        put_fleet_on_hold,
        update_fleet,
        get_actual_demand
    )
import rich
from scipy.stats import truncweibull_min


from gurobi_utils.license import load_wsl_lic

LICENSE_DICT = load_wsl_lic('gurobi.lic')

_demand, _datacenters, _servers, _selling_prices = load_problem_data(path="../../data")
# GET THE DEMAND
# SET THE RANDOM SEED
np.random.seed(3329)
actual_demand = get_actual_demand(_demand)

env = gp.Env(params=LICENSE_DICT)
env.setParam('OutputFlag', 0)
env.start()

def get_maintenance_cost(b, x=1, xhat=96):
    # CALCULATE THE CURRENT MAINTENANCE COST
    return int(b * (1 + (((1.5)*(x))/xhat * np.log2(((1.5)*(x))/xhat))))

# Define the failure rate function
def adjust_capacity_by_failure_rate(x):
    # Calculate the failure rate f using a truncated Weibull distribution
    failure_rate = truncweibull_min.rvs(0.3, 0.05, 0.1, size=1).item()
    # Adjust capacity by the failure rate
    return x * (1 - failure_rate)

# Define the sets (Servers, Time, Datacenters)
begin_ts = 0
time_steps = np.arange(begin_ts + 1, 168 + 1)
latency_sensitivity = ['high', 'medium', 'low']
datacenters_id = ['DC1', 'DC2', 'DC3', 'DC4']
server_types = ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3']
# Capacity of datacenters
datacenter_slots = {'DC1': 25245, 'DC2': 15300, 'DC3': 7020, 'DC4': 8280}
datacenter_slots = {k: v-2 for k, v in datacenter_slots.items()}

# Mapping between datacenters and latency sensitivity
datacenter_latency_map = {'DC1': 'low', 'DC2': 'medium', 'DC3': 'high', 'DC4': 'high'}
buying_cost = {'CPU.S1': 15000, 'CPU.S2': 16000, 'CPU.S3': 19500, 'CPU.S4': 22000, 'GPU.S1':120000, 'GPU.S2':140000, 'GPU.S3':160000}
moving_cost = 1000
energy_cost = {'DC1': 0.25, 'DC2': 0.35, 'DC3': 0.65, 'DC4':0.75}
maintain_cost = {'CPU.S1': 288, 'CPU.S2': 308, 'CPU.S3': 375, 'CPU.S4': 423, 'GPU.S1':2310, 'GPU.S2':2695, 'GPU.S3':3080}

server_slots = {'CPU.S1': 2, 'CPU.S2': 2, 'CPU.S3': 2, 'CPU.S4': 2, 'GPU.S1':4, 'GPU.S2':4, 'GPU.S3':4}
server_capacity = {'CPU.S1': 60, 'CPU.S2': 75, 'CPU.S3': 120, 'CPU.S4': 160, 'GPU.S1':8, 'GPU.S2':8, 'GPU.S3':8}
life_expectancy = 96  # Servers older than 96 time steps perish
lifespan = np.arange(1, life_expectancy + 1)
release_time = {'CPU.S1': [1,60], 'CPU.S2': [37,96], 'CPU.S3': [73,100], 'CPU.S4': [109,168], 'GPU.S1':[1,72], 'GPU.S2':[49,120], 'GPU.S3':[97,168]}
energy_consumption = {'CPU.S1': 400, 'CPU.S2': 460, 'CPU.S3': 800, 'CPU.S4': 920, 'GPU.S1':3000, 'GPU.S2':3000, 'GPU.S3':4200}

energy_prices = {('CPU.S1', 'DC1'): 100.0,
                ('CPU.S1', 'DC2'): 140.0,
                ('CPU.S1', 'DC3'): 260.0,
                ('CPU.S1', 'DC4'): 300.0,
                ('CPU.S2', 'DC1'): 115.0,
                ('CPU.S2', 'DC2'): 161.0,
                ('CPU.S2', 'DC3'): 299.0,
                ('CPU.S2', 'DC4'): 345.0,
                ('CPU.S3', 'DC1'): 200.0,
                ('CPU.S3', 'DC2'): 280.0,
                ('CPU.S3', 'DC3'): 520.0,
                ('CPU.S3', 'DC4'): 600.0,
                ('CPU.S4', 'DC1'): 230.0,
                ('CPU.S4', 'DC2'): 322.0,
                ('CPU.S4', 'DC3'): 598.0,
                ('CPU.S4', 'DC4'): 690.0,
                ('GPU.S1', 'DC1'): 750.0,
                ('GPU.S1', 'DC2'): 1050.0,
                ('GPU.S1', 'DC3'): 1950.0,
                ('GPU.S1', 'DC4'): 2250.0,
                ('GPU.S2', 'DC1'): 750.0,
                ('GPU.S2', 'DC2'): 1050.0,
                ('GPU.S2', 'DC3'): 1950.0,
                ('GPU.S2', 'DC4'): 2250.0,
                ('GPU.S3', 'DC1'): 1050.0,
                ('GPU.S3', 'DC2'): 1470.0,
                ('GPU.S3', 'DC3'): 2730.0,
                ('GPU.S3', 'DC4'): 3150.0
                }

demand = {}
D = actual_demand.copy()
for ts in range(begin_ts + 1, 168 + 1):    
    for sg in server_types:
        for ls in latency_sensitivity:
            value = D[(D['time_step'] == ts) & (D['server_generation'] == sg)][ls].values
            if value.size > 0:
                demand[(ts, sg, ls)] = value[0]
            else:
                demand[(ts, sg, ls)] = 0

hiring_prices = {}
for sg in server_types:
    for ls in latency_sensitivity:  
        hiring_prices[(sg, ls)] = _selling_prices[(_selling_prices['server_generation'] == sg) & (_selling_prices['latency_sensitivity'] == ls)]['selling_price'].values[0]

# with open(f'inventory_at_{begin_ts}.pkl', 'rb') as f:
#     initial_inventory = pickle.load(f)

# Create model
model = gp.Model('Cloud_Center_Optimization', env = env)
model.setParam('Seed', 42)

# Decision variables
# Decision variables
buy = model.addVars(time_steps, server_types, datacenters_id, lb = 0, vtype=GRB.CONTINUOUS, name="buy")  # Number of servers bought
# Move action
move_to = model.addVars(range(begin_ts, 168 + 1), server_types, range(1, life_expectancy + 1), datacenters_id, lb = 0, vtype=GRB.CONTINUOUS, name="move_to")  # Number of servers moved
remove_from = model.addVars(range(begin_ts, 168 + 1), server_types, range(1, life_expectancy + 1), datacenters_id, lb = 0, vtype=GRB.CONTINUOUS, name="remove_from")  # Number of servers moved

# Decision variables to track server lifespan in inventory
# I[t, s, l, life] represents the number of servers of type s with lifespan 'life' in latency group l at time t
I = model.addVars(range(begin_ts, 168 + 1), server_types, range(1, life_expectancy + 2), datacenters_id, lb = 0, vtype=GRB.CONTINUOUS, name="inventory")

# Auxiliary variable for min(I, demand)
min_var = model.addVars(time_steps, server_types, latency_sensitivity, vtype=GRB.CONTINUOUS,lb = 0, name="min_var")
Zf_var = model.addVars(time_steps, server_types, latency_sensitivity, vtype=GRB.CONTINUOUS,lb = 0, name="Zf_var")
diff_demand_capacity = model.addVars(time_steps, server_types, latency_sensitivity, vtype=GRB.CONTINUOUS,lb = 0, name="diff_demand_capacity")

# Loop over time steps to calculate total profit and define normalized lifespan constraints
total_profit = gp.LinExpr()
total_L = gp.LinExpr()
total_U = gp.LinExpr()
total_U1 = []
total_U2 = []
total_buy = gp.LinExpr()
total_L1 = gp.LinExpr()
total_L2 = gp.LinExpr()
total_diff_demand_capacity = gp.LinExpr()


for sg in server_types:
    for dc in datacenters_id:
        for life in range(1, life_expectancy + 2):
            model.addConstr(I[begin_ts, sg, life, dc] == 0, 
                            name=f"set_up_initial_inventory_{begin_ts}_{sg}_{dc}_life_{life}") 
            
# for sg in server_types:
    # for dc in datacenters_id:
    #     for life in range(1, life_expectancy + 2):
    #         model.addConstr(I[begin_ts, sg, life, dc] == initial_inventory[begin_ts, sg, life, dc], 
    #                         name=f"set_up_initial_inventory_{begin_ts}_{sg}_{dc}_life_{life}") 
            

for ts in time_steps:

    # 1 Buying constraint respecting the release window
    for sg in server_types:
        for dc in datacenters_id:
            # Only allow purchases during the server's release time window
            if (ts >= release_time[sg][0]) and (ts <= release_time[sg][1]):
                # model.addConstr(buy[ts, sg, dc] <= M * buy_decision[ts, sg, dc], name=f"buy_constraint_{ts}_{sg}_{dc}")
                model.addConstr(I[ts, sg, 1, dc] == buy[ts, sg, dc], name=f"buy_constraint_{ts}_{sg}_{dc}")
            else:
                # Prevent buying servers outside the release window
                model.addConstr(buy[ts, sg, dc] == 0, name=f"no_buy_{ts}_{sg}_{dc}")
                model.addConstr(I[ts, sg, 1, dc] == 0, name=f"buy_{ts}_{sg}_{dc}")
           
    # 2.1 Remove and move constraint: available resources
    for sg in server_types:
        for dc in datacenters_id:
            for life in range(2, life_expectancy + 1):
                model.addConstr(remove_from[ts, sg, life, dc] <= I[ts - 1, sg, life - 1, dc] + move_to[ts, sg, life, dc],
                name = f"remove_constraint_availabel_resource_{ts - 1}_{sg}_{life}_{dc}")
            model.addConstr(remove_from[ts, sg, 1, dc] == 0, name=f"remove_constraint_no_beginning_{begin_ts}_{sg}_{dc}")
            model.addConstr(move_to[ts, sg, 1, dc] == 0, name=f"move_constraint_no_beginning_{begin_ts}_{sg}_{dc}")


    # 2.2 Balance move_to and remove_from
    for sg in server_types:
        for life in range(2, life_expectancy + 1):
            model.addConstr(gp.quicksum(move_to[ts, sg, life, dc] for dc in datacenters_id)  == gp.quicksum(remove_from[ts, sg, life, dc] for dc in datacenters_id),
                                name=f"balance_move_to_remove_from_{ts}_{sg}_life_{life}")

    # 2.3 Update inventory after moving servers
    for sg in server_types:
        for dc in datacenters_id:
            for life in range(2, life_expectancy + 1):
                model.addConstr(I[ts, sg, life, dc] == I[ts - 1, sg, life - 1, dc] + move_to[ts, sg, life, dc] - remove_from[ts, sg, life, dc],
                                name=f"update_inventory_after_move_{ts}_{sg}_{dc}_life_{life}")
            # model.addConstr(I[ts, sg, 1, dc] == 0, name=f"update_inventory_no_buy_beginning_{begin_ts}_{sg}_{dc}")


    # 3. Constraint: Slot capacity of each datacenter
    for dc in datacenters_id:
        model.addConstr(
            gp.quicksum(I[ts, sg, life, dc] * server_slots[sg] for sg in server_types for life in range(1, life_expectancy + 1)) <= datacenter_slots[dc],
            name=f"slot_capacity_{ts}_{dc}"
        )

    
    # Add constraint to enforce min_var[t, s, l] = min(I[t, s, l], demand[t, s, l])
    for sg in server_types:
        for ls in latency_sensitivity:
            # Total inventory across all lifespans
            total_inventory = gp.quicksum(I[ts, sg, life, dc] for dc in datacenters_id if datacenter_latency_map[dc] == ls for life in range(1, life_expectancy + 1))
            capacity = total_inventory * server_capacity[sg]
            model.addConstr(Zf_var[ts, sg, ls] == adjust_capacity_by_failure_rate(capacity), name=f"custom_capacity_{ts}_{sg}_{ls}")
            model.addConstr(min_var[ts, sg, ls] <= Zf_var.get((ts, sg, ls), 0), name=f"min_capacity_{ts}_{sg}_{ls}")
            model.addConstr(min_var[ts, sg, ls] <= demand.get((ts, sg, ls), 0), name=f"min_demand_{ts}_{sg}_{ls}")

            
    # # Difference between demand and capacity
    # for sg in server_types:
    #     for ls in latency_sensitivity:
    #         model.addConstr(diff_demand_capacity[ts, sg, ls] >= demand.get((ts, sg, ls), 0) - Zf_var[ts, sg, ls], name=f"diff_demand_capacity_1_{ts}_{sg}_{ls}")
    #         model.addConstr(diff_demand_capacity[ts, sg, ls] >= Zf_var[ts, sg, ls] - demand.get((ts, sg, ls), 0), name=f"diff_demand_capacity_2_{ts}_{sg}_{ls}")
    #         model.addConstr(diff_demand_capacity[ts, sg, ls] <= demand.get((ts, sg, ls), 0), name=f"diff_demand_capacity_less_than_demand{ts}_{sg}_{ls}")

    # Revenue at time step t
    revenue_t = gp.quicksum(
        min_var[ts, sg, ls] * hiring_prices.get((sg, ls), 0)
        for sg in server_types for ls in latency_sensitivity
    )
    
    # Cost at time step t
    buy_cost_t = gp.quicksum(buying_cost[sg] * buy[ts, sg, dc] for sg in server_types for dc in datacenters_id)
    energy_cost_t = gp.quicksum(energy_prices[sg, dc] * gp.quicksum(I[ts, sg, life, dc] for life in range(1, life_expectancy + 1)) for sg in server_types for dc in datacenters_id)
    # **Maintenance cost calculation**:
    # Loop over each server and calculate the maintenance cost based on lifespan (life)

    maintenance_cost_t = gp.quicksum(
        get_maintenance_cost(maintain_cost[sg], life) * I[ts, sg, life, dc]
        for sg in server_types for dc in datacenters_id for life in range(1, life_expectancy + 1)
    )

    moving_cost_t = moving_cost * gp.quicksum(move_to[ts, sg, life, dc] for life in range(2, life_expectancy + 1) for sg in server_types for dc in datacenters_id )
    
    
    # Total cost at time step t
    total_cost_t = buy_cost_t + energy_cost_t + maintenance_cost_t + moving_cost_t
    
    # Profit at time step t
    profit_t = revenue_t - total_cost_t
    
    # Accumulate the total profit
    total_profit += profit_t
    
    demand_ts = 0
    for sg in server_types:
        for ls in latency_sensitivity:
            demand_ts += demand.get((ts, sg, ls), 0)
    L1 = (1/life_expectancy) * gp.quicksum([life * gp.quicksum(I[ts, sg, life, dc] for sg in server_types for dc in datacenters_id) for life in range(1, life_expectancy + 1)])
    L2 =  gp.quicksum([gp.quicksum(I[ts, sg, life, dc] for sg in server_types for dc in datacenters_id) for life in range(1, life_expectancy + 1)])
    total_L += (L1 - L2)
    total_L1 += L1
    total_L2 += L2
    # U = gp.quicksum(demand.get((ts, sg, ls), 0)- min_var[ts, sg, ls] for sg in server_types for ls in latency_sensitivity)
    IxG = actual_demand[actual_demand['time_step']==ts][['high','low','medium']].values
    IxG = IxG[IxG > 0]
    IxG = len(IxG)
    U1 = gp.quicksum(demand.get((ts, sg, ls), 0) for sg in server_types for ls in latency_sensitivity)
    U2 = gp.quicksum(min_var[ts, sg, ls] for sg in server_types for ls in latency_sensitivity)
    total_U1.append(U1)
    total_U2.append(U2)

    total_U += (U1 - U2) * 1 / IxG

    # Accumulate the total buying servers
    for sg in server_types:
        for dc in datacenters_id:
            total_buy += buy[ts, sg, dc]
    # # Accumulate the difference between demand and capacity
    # total_diff_demand_capacity += gp.quicksum(diff_demand_capacity[ts, sg, ls] for sg in server_types for ls in latency_sensitivity)

# model.ModelSense = GRB.MAXIMIZE
# model.setObjectiveN(-total_diff_demand_capacity, index = 0, priority = 1, weight = 1)
# model.setObjectiveN(total_profit, index = 1, priority = 1, weight = 1)

# model.setObjective(total_diff_demand_capacity, GRB.MINIMIZE)
# objective = total_profit + total_L - total_U
objective = total_profit 
model.setObjective(objective, GRB.MAXIMIZE)
model.update()
# model.write("model_move_only.lp")

# Gurobi optimization settings to improve speed and performance
opt = {
    "Presolve": 2,       # Aggressive presolve
    "PreSparsify": 2,    # Remove redundant constraints
    "Aggregate": 2,      # Aggressive aggregation
    "OutputFlag": 1,     # Enable output to track progress
    "MIPGap": 0.01       ## Allow 1% optimality gap
}

# Set model parameters
model.setParam('Presolve', opt["Presolve"])
model.setParam('PreSparsify', opt["PreSparsify"])
model.setParam('Aggregate', opt["Aggregate"])
model.setParam('OutputFlag', opt["OutputFlag"])
# model.setParam('MIPGap', opt["MIPGap"])
# Optimize model
model.optimize()
# model.write("model_move_only.ilp")  # This writes the IIS to a file for review

rich.print(f"Linear Objective: {model.objVal:0,.2f}")
rich.print(f"Profit: {total_profit.getValue():0,.2f}")
rich.print(f"Lifespan(Normalized): {96*total_L1.getValue()/total_L2.getValue():0,.2f}")
rich.print(f"Number of Servers: {total_buy.getValue():0,.2f}")
Uvalue=0
for ts in time_steps:
    Uvalue += (total_U2[ts-1].getValue() / total_U1[ts-1].getValue())
rich.print(f"Utilization: {Uvalue:0,.2f}")
# import pdb; pdb.set_trace()
# Extract and print results
if model.status == GRB.OPTIMAL:
    R = 0
    C = 0
    accumulated_profit = 0
    action_dict = {
        'server_generation': [],
        'datacenter_id': [],
        'time_step': [],
        'lifespan': [],
        'buying_servers': [],
        'remove': [],
        'move_to': [],
        'dismiss': []
    }
    for ts in range(begin_ts + 1,168 + 1):
        # print(f"\nTime step: {ts}")

        # Bought servers
        for sg in server_types:
            for dc in datacenters_id:
                bought_qty = buy[ts, sg, dc].x
                action_dict['server_generation'].append(sg)
                action_dict['datacenter_id'].append(dc)
                action_dict['time_step'].append(ts)
                action_dict['lifespan'].append(1)
                action_dict['buying_servers'].append(bought_qty)
                action_dict['remove'].append(0)
                action_dict['move_to'].append(0)
                action_dict['dismiss'].append(0)

                # Create solution dataframe
                # action_df = pd.concat([action_df, pd.DataFrame({
                #     'server_generation': [sg],
                #     'datacenter_id': [dc],
                #     'time_step': [ts],
                #     'lifespan': [1],
                #     'buying_servers': [bought_qty],
                #     'remove': [0],
                #     'move_to': [0],
                # })], ignore_index=True)
                # if bought_qty > 0:
                #     print(f"Buy {int(bought_qty)} {sg} at datacenter {dc}.")
        
        # Moving servers
        for sg in server_types:
            for dc in datacenters_id:
                for life in range(2, life_expectancy + 1):
                    move_qty = move_to[ts, sg, life, dc].x
                    # if move_qty > 0:
                    #     print(f"Move {int(move_qty)} {sg} with lifespan {life} to {dc}.")
                    # remove_qty = remove_from[ts, sg, life, dc].x
                    # if remove_qty > 0:
                    #     print(f"Remove {int(remove_qty)} {sg} with lifespan {life} from {dc}.")
                    # # Adding a row of action_df
                    remove_at = remove_from[ts, sg, life, dc].x
                    moveto_at = move_to[ts, sg, life, dc].x

                    action_dict['server_generation'].append(sg)
                    action_dict['datacenter_id'].append(dc)
                    action_dict['time_step'].append(ts)
                    action_dict['lifespan'].append(life)
                    action_dict['buying_servers'].append(0)
                    action_dict['remove'].append(remove_at)
                    action_dict['move_to'].append(moveto_at)
                    action_dict['dismiss'].append(0)
                    
                    # # Create solution dataframe
                    # action_df = pd.concat([action_df, pd.DataFrame({
                    #     'server_generation': [sg],
                    #     'datacenter_id': [dc],
                    #     'time_step': [ts],
                    #     'lifespan': [life],
                    #     'buying_servers': [0],
                    #     'remove': [remove_at],
                    #     'move_to': [moveto_at],
                    # })], ignore_index=True)

#         # Get Demand
#         demand_t = sum(demand.get((ts, sg, ls), 0) * hiring_prices.get((sg, ls), 0) for sg in server_types for ls in latency_sensitivity)
        
#         # Calculate revenue
#         revenue_t = sum(
#             min_var[ts, sg, ls].x * hiring_prices.get((sg, ls), 0)
#             for sg in server_types for ls in latency_sensitivity
#         )

#         R += revenue_t
        
#         # Calculate moving cost
#         moving_cost_t = moving_cost * sum(move_to[ts, sg, life, dc].x for sg in server_types for dc in datacenters_id for life in range(2, life_expectancy + 1))
        
        
#         # Calculate energy cost
#         energy_cost_t = sum(
#             energy_prices[sg, dc] * sum(I[ts, sg, life, dc].x for life in range(1, life_expectancy+1))
#                 for sg in server_types for dc in datacenters_id
#         )
        
#         # Calculate maintenance cost
#         maintenance_cost_t = sum(
#             get_maintenance_cost(maintain_cost[sg], life) * I[ts, sg, life, dc].x
#             for sg in server_types for dc in datacenters_id for life in range(1, life_expectancy+1)
#         )

#         # Total cost at time step `ts`
#         total_cost_t = moving_cost_t + energy_cost_t + maintenance_cost_t

#         C += total_cost_t

#         # Profit at time step `ts`
#         profit_t = revenue_t - total_cost_t
#         accumulated_profit += profit_t

#         # Print the calculated results for this time step
#         print(f" Time step {ts}:")
#         print(f" Revenue based on demand: {demand_t:0,.2f}")
#         print(f"  Revenue: {revenue_t:0,.2f}")
#         print(f"  Moving Cost: {moving_cost_t:0,.2f}")
#         print(f"  Energy Cost: {energy_cost_t:0,.2f}")
#         print(f"  Maintenance Cost: {maintenance_cost_t:0,.2f}")
#         print(f"  Total Cost: {total_cost_t:0,.2f}")
#         print(f"  Profit: {profit_t:0,.2f}\n")

#     rich.print(f"Profit: {accumulated_profit:0,.2f}")
#     rich.print(f"Cost: {C:0,.2f}")
#     rich.print(f"Revenue: {R:0,.2f}")
# else:
#     print("No optimal solution found.")
action_df = pd.DataFrame(action_dict)
action_df.to_csv('all_action_df.csv', index=False)
# df_to_submissions(initial_inventory, action_df)
