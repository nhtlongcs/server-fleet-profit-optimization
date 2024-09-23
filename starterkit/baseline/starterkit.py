
import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
from evaluation import get_actual_demand
from dotenv import load_dotenv
import os 
from pathlib import Path
load_dotenv()
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from utils import load_problem_data
import uuid
from evaluation import (
    change_selling_prices_format,
    get_time_step_demand,
    update_fleet,
    get_capacity_by_server_generation_latency_sensitivity,
    check_datacenter_slots_size_constraint,
    get_utilization,
    get_normalized_lifespan,
    get_profit,
    put_fleet_on_hold,
)


dataroot = Path(os.getenv('DATAROOT'))
assert dataroot.exists(), "DATAROOT must be a valid path to the data directory"
seeds = known_seeds('training')

strategy = 'buy_only'
output_dir = Path('./output') / strategy
output_dir.mkdir(exist_ok=True, parents=True)


# ----------------- HELPER FUNCTIONS ----------------- #

def _check_release_time(ts, rt):
    # HELPER FUNCTION TO CHECK THE CORRECT SERVER USAGE BY TIME-STEP
    rt = eval(rt)
    if ts >= min(rt) and ts <= max(rt):
        return True
    else:
        return False
    
def solution_wraper(solution, timestep, datacenters, servers, selling_prices):
    if len(solution) == 0:
        return ( 
        pd.DataFrame(columns=['action', 'server_id', 'datacenter_id', 'server_generation',
       'server_type', 'release_time', 'purchase_price', 'slots_size',
       'energy_consumption', 'capacity', 'life_expectancy', 'cost_of_moving',
       'average_maintenance_fee', 'cost_of_energy', 'latency_sensitivity',
       'slots_capacity', 'selling_price']), 
        pd.DataFrame(columns=['action', 'server_id', 'datacenter_id', 'server_generation',
       'server_type', 'release_time', 'purchase_price', 'slots_size',
       'energy_consumption', 'capacity', 'life_expectancy', 'cost_of_moving',
       'average_maintenance_fee', 'cost_of_energy', 'latency_sensitivity',
       'slots_capacity', 'selling_price', 'time_step'])
       )
        
    solution = pd.DataFrame(solution)
    solution = solution.merge(servers, on='server_generation', how='left')
    solution = solution.merge(datacenters, on='datacenter_id', how='left')
    solution = solution.merge(selling_prices, 
                              on=['server_generation', 'latency_sensitivity'], 
                              how='left')
    solution.reset_index(drop=True, inplace=False)
    batch = solution.copy()
    batch['time_step'] = timestep
    assert isinstance(solution, pd.DataFrame), 'solution must be a pandas DataFrame'
    s = solution
    s = s.drop_duplicates('server_id', inplace=False)
    s = s.set_index('server_id', drop=False, inplace=False)
    return s, batch

# ----------------- HELPER FUNCTIONS ----------------- #

# ----------------- YOUR CODE HERE ------------------- #

def solve(demand, datacenters, prices, _server_info, deployed_servers, ts):
    
    # Initialize some variables
    actions = []
    latency_sensitivities = ['low', 'medium', 'high']
    latency_datacenters = {
        'low': ['DC1'],
        'medium': ['DC2'],
        'high': ['DC3', 'DC4']
    }

    capicity_of_servers = {}
    for server_id, server in _server_info.items():
        capicity_of_servers[server_id] = {k: 0 for k in latency_sensitivities}
    quota_of_datacenters = {datacenter_id: datacenter['slots_capacity'] for datacenter_id, datacenter in datacenters.items()}

    
    # update server availability
    server_info = {}
    for server_id, server in _server_info.items():
        if _check_release_time(ts, server['release_time']):
            server_info[server_id] = server
    
    if not(deployed_servers.empty):
        # Remove old servers
        deployed_servers['lifespan'] = ts - deployed_servers['time_step']
        deployed_servers = deployed_servers.drop(deployed_servers.index[deployed_servers['lifespan'] >= deployed_servers['life_expectancy']], inplace=False)
        
        # update quota of datacenters
        slots = deployed_servers.groupby(by=['datacenter_id']).agg({'slots_size': 'sum','slots_capacity': 'mean'})
        quota_of_datacenters = {datacenter_id: datacenter['slots_capacity'] - slots.loc[datacenter_id, 'slots_size'] if datacenter_id in slots.index else datacenter['slots_capacity'] for datacenter_id, datacenter in datacenters.items()}
        
        # update capicity of servers
        z = get_capacity_by_server_generation_latency_sensitivity(deployed_servers)
        for server_id in z.index:
            for l in latency_sensitivities:
                capicity_of_servers[server_id][l] = z.loc[server_id][l]
            
    for l in latency_sensitivities:
        for sg, d in demand[l].items():
            server_generation = sg

            c = capicity_of_servers[server_generation][l] # capicity of current server generation
            if server_generation not in server_info:
                # unable to buy new server
                continue 
            if d > c:
                one_instance_cap = server_info[server_generation]['capacity']
                # num of instances to buy to meet the demand
                num = (d - c) // one_instance_cap + 1
                for datacenter_id in latency_datacenters[l]:
                    if num == 0: 
                        break # no need to buy more
                    if num > 0:
                        # buy servers
                        dc_slot_cap = quota_of_datacenters[datacenter_id]
                        sv_slot_size = server_info[server_generation]['slots_size']
                        
                        max_instance_of_center = dc_slot_cap // sv_slot_size
                        buy_num = int(min(num, max_instance_of_center))
                        if buy_num == 0:
                            continue

                        print(f'Buy {buy_num} server {server_generation} for latency sensitivity {l}')
                        
                        for _ in range(buy_num):
                            actions.append({
                                "action": "buy",
                                "server_id": str(uuid.uuid4()),
                                "datacenter_id": datacenter_id,
                                "server_generation": server_generation
                            })
                        
                        quota_of_datacenters[datacenter_id] -= sv_slot_size * buy_num
                        capicity_of_servers[server_generation][l] += one_instance_cap * buy_num
                        num -= buy_num
                    if num < 0:
                        raise ValueError('num should not be negative')
                if num == 0:
                    print(f'satisfied')
    return actions


# ----------------- YOUR CODE HERE ------------------- #

# ----------------- EVALUATION PIPELINE ------------------- #

def get_my_solution(demand):
    
    _, datacenters_df, server_info_df, selling_prices_df = load_problem_data(path=dataroot)
    datacenters = datacenters_df.set_index('datacenter_id').to_dict(orient='index')
    server_info = server_info_df.set_index('server_generation').to_dict(orient='index')

    solution = pd.DataFrame()
    selling_prices = change_selling_prices_format(selling_prices_df)

    OBJECTIVE = 0
    FLEET = pd.DataFrame()
    
    metrics = {
        'U': [],
        'L': [],
        'P': [],
        'O': [],
    }
    # if ts-related fleet is empty then current fleet is ts-fleet
    for ts in range(1, 168 + 1):
        print('Time step:', ts)
        # GET THE ACTUAL DEMAND AT TIMESTEP ts
        D = get_time_step_demand(demand, ts)
        sol = solve(D, datacenters, selling_prices, server_info, solution, ts)

        # GET THE SERVERS DEPLOYED AT TIMESTEP ts
        ts_fleet, batch_solution = solution_wraper(sol, ts, datacenters_df, server_info_df, selling_prices_df)
        # UPDATE SOLUTION
        solution = pd.concat([solution, batch_solution], ignore_index=True)
        if ts_fleet.empty:
            ts_fleet = FLEET

        # UPDATE FLEET
        FLEET = update_fleet(ts, FLEET, ts_fleet)

        # CHECK IF THE FLEET IS EMPTY
        if FLEET.shape[0] > 0:
            # GET THE SERVERS CAPACITY AT TIMESTEP ts
            Zf = get_capacity_by_server_generation_latency_sensitivity(FLEET)
    
            # CHECK CONSTRAINTS
            check_datacenter_slots_size_constraint(FLEET)
    
            # EVALUATE THE OBJECTIVE FUNCTION AT TIMESTEP ts
            U = get_utilization(D, Zf)
    
            L = get_normalized_lifespan(FLEET)
    
            P = get_profit(D, 
                           Zf, 
                           selling_prices,
                           FLEET)

            o = U * L * P
            OBJECTIVE += o

            metrics['U'].append(U)
            metrics['L'].append(L)
            metrics['P'].append(P)
            metrics['O'].append(OBJECTIVE)
            
            # PUT ENTIRE FLEET on HOLD ACTION
            FLEET = put_fleet_on_hold(FLEET)

            # PREPARE OUTPUT
            output = {'time-step': ts,
                      'O': round(OBJECTIVE, 2),
                      'U': round(U, 2),
                      'L': round(L, 2),
                      'P': round(P, 2)}
        else:
            # PREPARE OUTPUT
            output = {'time-step': ts,
                      'O': np.nan,
                      'U': np.nan,
                      'L': np.nan,
                      'P': np.nan}

    solution = solution[['time_step', 'action', 'server_id', 'datacenter_id', 'server_generation']]
    solution = solution.to_dict(orient='records')

    print(f'Utilization: {sum(metrics["U"]):0,.2f}')
    print(f'Lifespan: {sum(metrics["L"]):0,.2f}')
    print(f'Profit: {sum(metrics["P"]):0,.2f}')
    print(f'Solution score: {OBJECTIVE:0,.2f}')

    return solution, metrics

# ----------------- EVALUATION PIPELINE ------------------- #

# -------------------- MAIN FUNCTION ------------------- #

if __name__ == '__main__':

    demand = pd.read_csv(dataroot / 'demand.csv')
    for seed in seeds:
        # SET THE RANDOM SEED
        np.random.seed(seed)

        # GET THE DEMAND
        actual_demand = get_actual_demand(demand)
        # CALL YOUR APPROACH HERE
        solution, metrics = get_my_solution(actual_demand)

        # SAVE YOUR SOLUTION
        save_solution(solution, output_dir / f'{seed}.json')

        X = list(range(len(metrics['O'])))
        for metric, values in metrics.items():
            plt.plot(X, values, label=metric)
            plt.xlabel('Time')
            plt.ylabel(metric)
            plt.legend()
            plt.savefig(output_dir /f'{metric}.png')
            plt.clf()

        break