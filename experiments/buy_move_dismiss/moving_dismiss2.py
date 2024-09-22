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

from baseline.evaluation import (
    get_capacity_by_server_generation_latency_sensitivity,
    check_datacenter_slots_size_constraint,
    put_fleet_on_hold,
    update_fleet,
    get_actual_demand
)
import rich
from scipy.stats import truncweibull_min
from dotenv import load_dotenv
load_dotenv()

if len(sys.argv) < 2:
    print('Usage: python moving_dismiss2.py <seed> <slot_bound>')
    raise SystemExit
else:
    seed = sys.argv[1]
    slot_bound = int(sys.argv[2])

from gurobi_utils.license import load_wsl_lic

license_path = os.environ.get('GUROBI_LIC_PATH')
assert license_path is not None, "Please set the GUROBI_LIC_PATH environment variable"

data_path = os.environ.get('DATAROOT')
assert data_path is not None, "Please set the DATAROOT environment variable"

LICENSE_DICT = load_wsl_lic(license_path)

_demand, datacenters, servers, _selling_prices, elasticity = load_problem_data(path=data_path)

# GET THE DEMAND
# SET THE RANDOM SEED
np.random.seed(int(seed))
actual_demand = get_actual_demand(_demand)

env = gp.Env(params=LICENSE_DICT)
env.setParam('OutputFlag', 0)
env.start()

def get_new_demand_for_new_price(d0, p0, p1, e):
    # CALCULATE THE NEW DEMAND ACCORDING TO THE NEW PRICE
    delta_p = (p1 - p0) / p0
    delta_p_e = delta_p * e
    d1 = d0 * (1 + delta_p_e)
    return d1

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

elasticity = elasticity.set_index(['server_generation', 'latency_sensitivity']).to_dict()['elasticity']

latency_sensitivity = list(datacenters['latency_sensitivity'].unique())
datacenters_id = list(datacenters['datacenter_id'].values)
datacenter_slots = datacenters.set_index('datacenter_id').to_dict()['slots_capacity']
# datacenter_slots = {k: v-2 for k, v in datacenter_slots.items()}
# datacenter_slots = {k: v-41 for k, v in datacenter_slots.items()}
datacenter_slots = {k: v-slot_bound for k, v in datacenter_slots.items()}

# datacenter_slots['DC3'] -= 20

server_types = list(servers['server_generation'].values)
buying_cost = servers.set_index('server_generation').to_dict()['purchase_price']

# moving_cost = servers.set_index('server_generation').to_dict()['cost_of_moving']
moving_cost = 1000
maintain_cost = servers.set_index('server_generation').to_dict()['average_maintenance_fee']
energy_cost = datacenters.set_index('datacenter_id').to_dict()['cost_of_energy']
energy_consumption = servers.set_index('server_generation').to_dict()['energy_consumption']
energy_prices = {(sg, dc): energy_cost[dc] * energy_consumption[sg] for sg in server_types for dc in datacenters_id}

server_slots = servers.set_index('server_generation').to_dict()['slots_size']
server_capacity = servers.set_index('server_generation').to_dict()['capacity']


datacenter_latency_map = datacenters.set_index('datacenter_id').to_dict()['latency_sensitivity']
# life_expectancy = servers.set_index('server_generation').to_dict()['life_expectancy']
life_expectancy = 96

release_time = servers.set_index('server_generation').to_dict()['release_time']
release_time = {k: eval(v) for k, v in release_time.items()}

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

with open(f'min_recorded_{seed}.pkl', 'rb') as f:
    min_recorded = pickle.load(f)
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
#Dismissing servers
dismiss = model.addVars(range(begin_ts, 168 + 1), server_types, range(2, life_expectancy + 1), datacenters_id, lb = 0, vtype=GRB.CONTINUOUS, name="dismiss")  # Number of servers moved


price_vars = model.addVars(time_steps, server_types, latency_sensitivity, lb=0, vtype=GRB.CONTINUOUS, name="price_vars")
demand_vars = model.addVars(time_steps, server_types, latency_sensitivity, lb=0, vtype=GRB.CONTINUOUS, name="demand_vars")

# Decision variables to track server lifespan in inventory
# I[t, s, l, life] represents the number of servers of type s with lifespan 'life' in latency group l at time t
I = model.addVars(range(begin_ts, 168 + 1), server_types, range(1, life_expectancy + 2), datacenters_id, lb = 0, vtype=GRB.CONTINUOUS, name="inventory")

# Auxiliary variable for min(I, demand)
min_var = model.addVars(time_steps, server_types, latency_sensitivity, vtype=GRB.CONTINUOUS,lb = 0, name="min_var")
Zf_var = model.addVars(time_steps, server_types, latency_sensitivity, vtype=GRB.CONTINUOUS,lb = 0, name="Zf_var")

demand_recorded = {ts: {sg: {ls: 0 for ls in latency_sensitivity} for sg in server_types} for ts in range(1,169)}


# Loop over time steps to calculate total profit and define normalized lifespan constraints
total_profit = gp.LinExpr()
total_L = gp.LinExpr()
total_U = gp.LinExpr()
total_buy = gp.LinExpr()

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
D_max = 2878969
dismiss_lb = 2
            
P_recorded = [0] * 169
U_recorded = [0] * 169
diff_recorded = 0
Zf_recorded = {ts: {sg: {ls: 0 for ls in latency_sensitivity} for sg in server_types} for ts in range(1,169)}

for ts in time_steps:
    # 0 Pricing Strategy Constraint 

    for sg in server_types:
        for ls in latency_sensitivity:
            price_at_ts = hiring_prices[(sg, ls)]
            e_at_ts = elasticity[(sg, ls)]
            demand_vars[ts, sg, ls] = get_new_demand_for_new_price(demand.get((ts, sg, ls), 0), price_at_ts, price_vars[ts, sg, ls], e_at_ts)
            model.addConstr(demand_vars[ts, sg, ls] >= 0, name=f"demand_constraint_{ts}_{sg}_{ls}")

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
                model.addConstr(remove_from[ts, sg, life, dc] <= I[ts - 1, sg, life - 1, dc] - dismiss[ts, sg, life, dc], # Dismiss before moving
                name = f"remove_constraint_available_resource_{ts - 1}_{sg}_{life}_{dc}")
            model.addConstr(remove_from[ts, sg, 1, dc] == 0, name=f"remove_constraint_no_beginning_{begin_ts}_{sg}_{dc}")
            model.addConstr(move_to[ts, sg, 1, dc] == 0, name=f"move_constraint_no_beginning_{begin_ts}_{sg}_{dc}")


    # 2.2 Balance move_to and remove_from
    for sg in server_types:
        for life in range(2, life_expectancy + 1):
            model.addConstr(gp.quicksum(move_to[ts, sg, life, dc] for dc in datacenters_id)  == gp.quicksum(remove_from[ts, sg, life, dc] for dc in datacenters_id),
                                name=f"balance_move_to_remove_from_{ts}_{sg}_life_{life}")
            
    # Dismiss servers constraint
    # if ts >= life_expectancy + 1:
    for sg in server_types:
        for dc in datacenters_id:
            for life in range(dismiss_lb, life_expectancy + 1):
                model.addConstr(dismiss[ts, sg, life, dc] <= I[ts - 1, sg, life - 1, dc], name=f"dismiss_constraint_{ts}_{sg}_{dc}_life_{life}")
            for life in range(2, dismiss_lb):
                model.addConstr(dismiss[ts, sg, life, dc] == 0, name=f"dismiss_constraint_{ts}_{sg}_{dc}_life_{life}")


    # 2.3 Update inventory after moving servers
    for sg in server_types:
            for dc in datacenters_id:
                for life in range(2, life_expectancy + 1):
                    model.addConstr(I[ts, sg, life, dc] == I[ts - 1, sg, life - 1, dc] + move_to[ts, sg, life, dc] - remove_from[ts, sg, life, dc] - dismiss[ts, sg, life, dc],
                            name=f"update_inventory_after_all_actions_{ts}_{sg}_{dc}_life_{life}")


    

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
            model.addConstr(min_var[ts, sg, ls] <= demand_vars[ts, sg, ls], name=f"min_demand_{ts}_{sg}_{ls}")

            model.addConstr(min_var[ts, sg, ls] == min_recorded[ts][sg][ls], name=f"min_recorded_capacity_{ts}_{sg}_{ls}")
            
    # # Difference between demand and capacity
    # for sg in server_types:
    #     for ls in latency_sensitivity:
    #         model.addConstr(diff_demand_capacity[ts, sg, ls] >= demand.get((ts, sg, ls), 0) - Zf_var[ts, sg, ls], name=f"diff_demand_capacity_1_{ts}_{sg}_{ls}")
    #         model.addConstr(diff_demand_capacity[ts, sg, ls] >= Zf_var[ts, sg, ls] - demand.get((ts, sg, ls), 0), name=f"diff_demand_capacity_2_{ts}_{sg}_{ls}")
    #         model.addConstr(diff_demand_capacity[ts, sg, ls] <= demand.get((ts, sg, ls), 0), name=f"diff_demand_capacity_less_than_demand{ts}_{sg}_{ls}")

    # Revenue at time step t
        # 1/2 * ((min_var[ts, sg, ls] * hiring_prices.get((sg, ls), 0)) + (min_recorded[int(ts)][sg][ls] * price_vars[ts, sg, ls]))
    revenue_t = gp.quicksum(
        # (min_var[ts, sg, ls] * hiring_prices.get((sg, ls), 0))
        min_recorded[ts][sg][ls] * price_vars[ts, sg, ls]
        for sg in server_types for ls in latency_sensitivity
    )

    for sg in server_types:
        for ls in latency_sensitivity:
            min_recorded[ts][sg][ls] = min_var[ts, sg, ls]
            Zf_recorded[ts][sg][ls] = Zf_var[ts, sg, ls]
       
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
    
    P_recorded[ts] = profit_t
    # Calculate demand at time step t
    L = (1/life_expectancy) * gp.quicksum([life * gp.quicksum(I[ts, sg, life, dc] for sg in server_types for dc in datacenters_id) for life in range(1, life_expectancy + 1)])
    total_L += L 


    U = gp.quicksum(min_var[ts, sg, ls] - Zf_var[ts, sg, ls] for sg in server_types for ls in latency_sensitivity)
    U_recorded[ts] = U
    total_U += U

    # Accumulate the total buying servers
    for sg in server_types:
        for dc in datacenters_id:
            total_buy += buy[ts, sg, dc]
    # Accumulate the total profit
    total_profit += profit_t

# objective = total_profit + 1700*total_L + 20*total_U
objective = total_profit

    # # Accumulate the difference between demand and capacity
    # total_diff_demand_capacity += gp.quicksum(diff_demand_capacity[ts, sg, ls] for sg in server_types for ls in latency_sensitivity)

# model.ModelSense = GRB.MAXIMIZE
# model.setObjectiveN(-total_diff_demand_capacity, index = 0, priority = 1, weight = 1)
# model.setObjectiveN(total_profit, index = 1, priority = 1, weight = 1)

# model.setObjective(total_diff_demand_capacity, GRB.MINIMIZE)

model.setObjective(objective, GRB.MAXIMIZE)
model.update()
# model.write("model_move_only.lp")

# Gurobi optimization settings to improve speed and performance
opt = {
    "Presolve": 2,       # Aggressive presolve
    "PreSparsify": 2,    # Remove redundant constraints
    "Aggregate": 2,      # Aggressive aggregation
    "OutputFlag": 1,      # Enable output to track progress
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
# rich.print(f"Lifespan: {total_L.getValue():0,.2f}")
# rich.print(f"Lifespan(Normalized): {total_L.getValue()/total_buy.getValue():0,.2f}")
# rich.print(f"Number of Servers: {total_buy.getValue():0,.2f}")
# rich.print(f"Utilize: {total_U.getValue():0,.2f}")

# import pdb; pdb.set_trace()
# Extract and print results
if model.status == GRB.OPTIMAL:
    R = 0
    C = 0
    detect_ts_less_than = []
    detect_ts_greater_than = []
    detect_ts_equal_to = []
    
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
        P_recorded[ts] = float(P_recorded[ts].getValue())
        U_recorded[ts] = float(U_recorded[ts].getValue())
        # Bought servers
        for sg in server_types:
            for ls in latency_sensitivity:
                demand_recorded[ts][sg][ls] = float(demand_vars[ts, sg, ls].getValue())
                Zf_recorded[ts][sg][ls] = float(Zf_recorded[ts][sg][ls].x)
                diff_recorded += Zf_recorded[ts][sg][ls] - demand_recorded[ts][sg][ls]
                if Zf_recorded[ts][sg][ls] == demand_recorded[ts][sg][ls]:
                    detect_ts_equal_to.append(ts)
                elif Zf_recorded[ts][sg][ls] > demand_recorded[ts][sg][ls]:
                    detect_ts_greater_than.append(ts)
                elif Zf_recorded[ts][sg][ls] < demand_recorded[ts][sg][ls]:
                    detect_ts_less_than.append(ts)    
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

                    
                    dismiss_qty = dismiss[ts, sg, life, dc].x
                    action_dict['dismiss'].append(dismiss_qty)
                    # if dismiss_qty > 0:
                    #     print(f"Dismiss {int(dismiss_qty)} {sg} with lifespan {life} from {dc}.")
    pricing_stategies = []
    for ts in range(begin_ts + 1,168 + 1):
        for sg in server_types:
            for ls in latency_sensitivity:
                pricing_stategy = {}
                pricing_stategy['time_step'] = ts
                pricing_stategy['server_generation'] = sg
                pricing_stategy['latency_sensitivity'] = ls
                pricing_stategy['price'] = price_vars[ts, sg, ls].x
                pricing_stategies.append(pricing_stategy)
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
# with open(f'P_recorded.json', 'w') as f:
#     json.dump(P_recorded, f)
# with open(f'U_recorded.json', 'w') as f:
#     json.dump(U_recorded, f)
with open(f'pricing_strategy_{seed}.json', 'w') as f:
    json.dump(pricing_stategies, f)

# with open(f'detect_ts_equal_to.pkl', 'wb') as f:
#     pickle.dump(detect_ts_equal_to, f)

# with open(f'detect_ts_less_than.pkl', 'wb') as f:
#     pickle.dump(detect_ts_less_than, f)

# with open(f'detect_ts_greater_than.pkl', 'wb') as f:
#     pickle.dump(detect_ts_greater_than, f)

# with open(f'new_demand.pkl', 'wb') as f:
#     pickle.dump(demand_recorded, f)
rich.print(f"Diff_U: {diff_recorded:0,.2f}")

# action_df = pd.DataFrame(action_dict)
# action_df.to_csv(f'all_action_df_{seed}.csv', index=False)
# action_df.to_csv('dismiss_action_df_reduce_slot.csv', index=False)
# import pdb; pdb.set_trace()
# df_to_submissions(initial_inventory, action_df)
