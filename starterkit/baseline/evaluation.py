

import logging
import numpy as np
import pandas as pd
from scipy.stats import truncweibull_min


# CREATE LOGGER
logger = logging.getLogger()
file_handler = logging.FileHandler('logs.log')
logger.addHandler(file_handler)

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
file_handler.setFormatter(formatter)


def get_known(key):
    # STORE SOME CONFIGURATION VARIABLES
    if key == 'datacenter_id':
        return ['DC1', 
                'DC2', 
                'DC3', 
                'DC4']
    elif key == 'actions':
        return ['buy',
                'hold',
                'move',
                'dismiss']
    elif key == 'server_generation':
        return ['CPU.S1', 
                'CPU.S2', 
                'CPU.S3', 
                'CPU.S4', 
                'GPU.S1', 
                'GPU.S2', 
                'GPU.S3']
    elif key == 'latency_sensitivity':
        return ['high', 
                'medium', 
                'low']
    elif key == 'required_columns':
        return ['time_step', 
                'datacenter_id', 
                'server_generation', 
                'server_id',
                'action']
    elif key == 'time_steps':
        return 168
    elif key == 'datacenter_fields':
        return ['datacenter_id', 
                'cost_of_energy',
                'latency_sensitivity', 
                'slots_capacity']
    elif key == 'price_strategy_columns':
        return ['time_step',
                'latency_sensitivity',
                'server_generation',
                'price']


def fleet_data_preparation(solution, servers, datacenters, selling_prices):
    # CHECK DATA FORMAT
    solution = check_data_format(solution)
    solution = check_actions(solution)
    # CHECK DATACENTERS AND SERVERS NAMES
    solution = check_datacenters_servers_generation(solution)
    # ADD PROBLEM DATA
    solution = solution.merge(servers, on='server_generation', how='left')
    solution = solution.merge(datacenters, on='datacenter_id', how='left')
    solution = solution.merge(selling_prices, 
                              on=['server_generation', 'latency_sensitivity'], 
                              how='left')
    # CHECK IF SERVERS ARE USED AT THE RIGHT RELEASE TIME
    solution = check_server_usage_by_release_time(solution)
    # DROP DUPLICATE SERVERS IDs
    solution = drop_duplicate_server_ids(solution)
    return solution.reset_index(drop=True, inplace=False)


def check_data_format(solution):
    # CHECK THAT WE HAVE ALL AND ONLY THE REQUIRED COLUMNS
    required_cols = get_known('required_columns')
    try:
        return solution[required_cols]
    except Exception:
        raise(ValueError('Please check the solution format.'))


def check_actions(solution):
    # CHECK THAT WE ARE USING ONLY ALLOWED ACTIONS
    actions = get_known('actions')
    solution = solution[solution['action'].isin(actions)]
    if not (solution[solution['time_step'] == 1]['action'] == 'buy').all():
        raise(ValueError('At time-step 1 it is only possible to use the "buy" action.'))
    return solution.reset_index(drop=True, inplace=False)


def check_datacenters_servers_generation(solution):
    # CHECK THAT DATA-CENTERS AND SERVER GENERATIONS ARE NAMED AS REQUESTED
    known_datacenters = get_known('datacenter_id')
    known_generations = get_known('server_generation')
    solution = solution[solution['datacenter_id'].isin(known_datacenters)]
    solution = solution[solution['server_generation'].isin(known_generations)]
    return solution


def check_server_usage_by_release_time(solution):
    # CHECK THAT ONLY THE SERVERS AVAILABLE FOR PURCHASE AT A CERTAIN TIME-STEP
    # ARE USED AT THAT TIME-STEP
    solution['rt_is_fine'] = solution.apply(check_release_time, axis=1)
    solution = solution[(solution['rt_is_fine'] != 'buy') | solution['rt_is_fine']]
    solution = solution.drop(columns='rt_is_fine', inplace=False)
    return solution


def check_release_time(x):
    # HELPER FUNCTION TO CHECK THE CORRECT SERVER USAGE BY TIME-STEP
    rt = eval(x['release_time'])
    ts = x['time_step']
    if ts >= min(rt) and ts <= max(rt):
        return True
    else:
        return False


def drop_duplicate_server_ids(solution):
    # DROP SERVERS THAT ARE BOUGHT MULTIPLE TIMES WITH THE SAME SERVER ID
    drop = solution[(solution['server_id'].duplicated()) & (solution['action'] == 'buy')].index
    if drop.any():
        solution = solution.drop(index=drop, inplace=False)
    return solution


def pricing_data_preparation(prices):
    # IF THERE IS NO PRICING STRATEGY DO NOTHING
    if prices.empty:
        return prices
    # CHECK DATA FORMAT
    required_cols = get_known('price_strategy_columns')
    try:
        prices = prices[required_cols]
    except Exception:
        raise(ValueError('Please check the price strategy solution format.'))
    # CHECK THERE IS ONLY 1 PRICE PER TIME-STEP PER LATENCY SENSITIVITY
    # AND SERVER GENERATION
    prices = prices.drop_duplicates(['time_step',
                                     'latency_sensitivity',
                                     'server_generation'],
                                     inplace=False,
                                     ignore_index=True)
    prices = prices[prices['price'] >= 0].copy()
    return prices


def change_elasticity_format(elasticity):
    # ADJUST THE FORMAT OF THE ELASTICITY DATAFRAME TO GET ALONG WITH THE
    # REST OF CODE
    elasticity = elasticity.pivot(index='server_generation', columns='latency_sensitivity')
    elasticity.columns = elasticity.columns.droplevel(0)
    return elasticity


def change_selling_prices_format(selling_prices):
    # ADJUST THE FORMAT OF THE SELLING PRICES DATAFRAME TO GET ALONG WITH THE
    # REST OF CODE
    selling_prices = selling_prices.pivot(index='server_generation', columns='latency_sensitivity')
    selling_prices.columns = selling_prices.columns.droplevel(0)
    return selling_prices


def get_actual_demand(demand):
    # CALCULATE THE ACTUAL DEMAND AT TIME-STEP t
    actual_demand = []
    for ls in get_known('latency_sensitivity'):
        for sg in get_known('server_generation'):
            d = demand[demand['latency_sensitivity'] == ls]
            sg_demand = d[sg].values.astype(float)
            rw = get_random_walk(sg_demand.shape[0], 0, 2)
            sg_demand += (rw * sg_demand)

            ls_sg_demand = pd.DataFrame()
            ls_sg_demand['time_step'] = d['time_step']
            ls_sg_demand['server_generation'] = sg
            ls_sg_demand['latency_sensitivity'] = ls
            ls_sg_demand['demand'] = sg_demand.astype(int)
            actual_demand.append(ls_sg_demand)

    actual_demand = pd.concat(actual_demand, axis=0, ignore_index=True)
    actual_demand = actual_demand.pivot(index=['time_step', 'server_generation'], columns='latency_sensitivity')
    actual_demand.columns = actual_demand.columns.droplevel(0)
    actual_demand = actual_demand.loc[actual_demand[get_known('latency_sensitivity')].sum(axis=1) > 0]
    actual_demand = actual_demand.reset_index(['time_step', 'server_generation'], col_level=1, inplace=False)
    return actual_demand


def get_random_walk(n, mu, sigma):
    # HELPER FUNCTION TO GET A RANDOM WALK TO CHANGE THE DEMAND PATTERN
    r = np.random.normal(mu, sigma, n)
    ts = np.empty(n)
    ts[0] = r[0]
    for i in range(1, n):
        ts[i] = ts[i - 1] + r[i]
    ts = (2 * (ts - ts.min()) / np.ptp(ts)) - 1
    return ts


def get_time_step_demand(demand, ts):
    # GET THE DEMAND AT A SPECIFIC TIME-STEP t
    d = demand[demand['time_step'] == ts]
    d = d.set_index('server_generation', drop=True, inplace=False)
    d = d.drop(columns='time_step', inplace=False).astype('float')
    return d


def get_time_step_fleet(solution, ts):
    # GET THE FLEET AT A SPECIFIC TIME-STEP 
    if ts in solution['time_step'].values:
        s = solution[solution['time_step'] == ts]
        s = s.drop_duplicates('server_id', inplace=False)
        s = s.set_index('server_id', drop=False, inplace=False)
        s = s.drop(columns='time_step', inplace=False)
        return s
    else:
        return pd.DataFrame()


def get_time_step_prices(pricing_strategy, ts):
    # GET THE PRICES AT A SPECIFIC TIME-STEP 
    if ts in pricing_strategy['time_step'].values:
        s = pricing_strategy[pricing_strategy['time_step'] == ts]
        s = s.drop(columns='time_step', inplace=False)
        s = s.pivot(index='server_generation', columns='latency_sensitivity')
        s.columns = s.columns.droplevel(0)
        return s
    else:
        return pd.DataFrame()


def update_selling_prices(selling_prices, ts_prices):
    # UPDATE THE SELLING PRICES ACCORDING TO THE PRICING STRATEGY
    if ts_prices.empty:
        return selling_prices
    else:
        selling_prices.update(ts_prices)
        return selling_prices


def update_demand_according_to_prices(D, selling_prices, base_prices, elasticity):
    # UPDATE THE DEMAND ACCORDING TO THE NEW PRICES
    new_prices = selling_prices.ne(base_prices)
    ix = new_prices.where(selling_prices.ne(base_prices)).stack().index.tolist()
    if ix:
        SG = D.index.values
        for sg, ls in ix:
            if sg in SG:
                d0 = D.loc[sg, ls]
                p0 = base_prices.loc[sg, ls]
                p1 = selling_prices.loc[sg, ls]
                e = elasticity.loc[sg, ls]
                d1 = get_new_demand_for_new_price(d0, p0, p1, e)
                D.loc[sg, ls] = d1
    return D


def get_new_demand_for_new_price(d0, p0, p1, e):
    # CALCULATE THE NEW DEMAND ACCORDING TO THE NEW PRICE
    delta_p = (p1 - p0) / p0
    delta_p_e = delta_p * e
    d1 = d0 * (1 + delta_p_e)
    if d1 < 0:
        return 0
    return int(d1)


def get_capacity_by_server_generation_latency_sensitivity(fleet):
    # CALCULATE THE CAPACITY AT A SPECIFIC TIME-STEP t FOR ALL PAIRS OF
    # LATENCY SENSITIVITIES AND SERVER GENERATIONS. ADJUST SUCH CAPACITY
    # ACCORDING TO THE FAILURE RATE f.
    Z = fleet.groupby(by=['server_generation', 'latency_sensitivity'])['capacity'].sum().unstack()
    cols = get_valid_columns(Z.columns, get_known('latency_sensitivity'))
    Z = Z[cols]
    Z = Z.map(adjust_capacity_by_failure_rate, na_action='ignore')
    Z = Z.fillna(0, inplace=False)
    return Z


def get_valid_columns(cols1, cols2):
    # HELPER FUNCTION TO GET THE COLUMNS THAT ARE IN THE DATAFRAME
    return list(set(cols1).intersection(set(cols2)))


def adjust_capacity_by_failure_rate(x):
    # HELPER FUNCTION TO CALCULATE THE FAILURE RATE f
    return int(x * (1 - truncweibull_min.rvs(0.3, 0.05, 0.1, size=1).item()))


def check_datacenter_slots_size_constraint(fleet):
    # CHECK DATACENTERS SLOTS SIZE CONSTRAINT
    slots = fleet.groupby(by=['datacenter_id']).agg({'slots_size': 'sum',
                                                     'slots_capacity': 'mean'})
    test = slots['slots_size'] > slots['slots_capacity']
    constraint = test.any()
    if constraint:
        print('Slots size')
        print(slots['slots_size'])
        print('Slots capacity')
        print(slots['slots_capacity'])

        raise(ValueError('Constraint 2 has been violated.'))
    return slots

def get_utilization(D, Z):
    # CALCULATE OBJECTIVE U = UTILIZATION
    u = []
    server_generations = Z.index
    latency_sensitivities = Z.columns
    for server_generation in server_generations:
        for latency_sensitivity in latency_sensitivities:
            z_ig = Z[latency_sensitivity].get(server_generation, default=0)
            d_ig = D[latency_sensitivity].get(server_generation, default=0)
            if (z_ig > 0) and (d_ig > 0):
                u.append(min(z_ig, d_ig) / z_ig)
            elif (z_ig == 0) and (d_ig == 0):
                continue
            elif (z_ig > 0) and (d_ig == 0):
                u.append(0)
            elif (z_ig == 0) and (d_ig > 0):
                continue
    if u:
        return sum(u) / len(u)
    else:
        return 0


def get_normalized_lifespan(fleet):
    # CALCULATE OBJECTIVE L = NORMALIZED LIFESPAN
    return (fleet['lifespan'] / fleet['life_expectancy']).sum() / fleet.shape[0]


def get_profit(D, Z, selling_prices, fleet):
    # CALCULATE OBJECTIVE P = PROFIT
    R = get_revenue(D, Z, selling_prices)
    C = get_cost(fleet)
    return R - C, R, C


def get_revenue(D, Z, selling_prices):
    # CALCULATE THE REVENUE
    r = 0
    server_generations = Z.index
    latency_sensitivities = Z.columns
    for server_generation in server_generations:
        for latency_sensitivity in latency_sensitivities:
            z_ig = Z[latency_sensitivity].get(server_generation, default=0)
            d_ig = D[latency_sensitivity].get(server_generation, default=0)
            p_ig = selling_prices[latency_sensitivity].get(server_generation, default=0)
            r += min(z_ig, d_ig) * p_ig
    return r


def get_cost(fleet):
    # CALCULATE THE SERVER COST - PART 1
    fleet['cost'] = fleet.apply(calculate_server_cost, axis=1)
    return fleet['cost'].sum()


def calculate_server_cost(row):
    # CALCULATE THE SERVER COST - PART 2
    c = 0
    r = row['purchase_price']
    b = row['average_maintenance_fee']
    x = row['lifespan']
    xhat = row['life_expectancy']
    e = row['energy_consumption'] * row['cost_of_energy']
    c += e
    alpha_x = get_maintenance_cost(b, x, xhat)
    c += alpha_x
    if x == 1:
        c += r
    elif row['moved'] == 1:
        c += row['cost_of_moving']
    return c


def get_maintenance_cost(b, x, xhat):
    # CALCULATE THE CURRENT MAINTENANCE COST
    return b * (1 + (((1.5)*(x))/xhat * np.log2(((1.5)*(x))/xhat)))


def update_fleet(ts, fleet, solution):
    # UPADATE THE FLEET ACCORDING TO THE ACTIONS AT THE CURRENT TIMESTEP
    if fleet.empty:
        fleet = solution.copy()
        fleet['lifespan'] = 0
        fleet['moved'] = 0
    else:
        server_id_action = solution[['action', 'server_id']].groupby('action')['server_id'].apply(list).to_dict()
        # BUY
        if 'buy' in server_id_action:
            fleet = pd.concat([fleet, solution[solution['action'] == 'buy']], axis=0)
        # MOVE
        if 'move' in server_id_action:
            s = server_id_action['move']
            dc_fields = get_known('datacenter_fields')
            fleet.loc[s, dc_fields] = solution.loc[s, dc_fields]
            fleet.loc[s, 'selling_price'] = solution.loc[s, 'selling_price']
            fleet.loc[s, 'moved'] = 1
        # HOLD
            # do nothing
        # DISMISS
        if 'dismiss' in server_id_action:
            fleet = fleet.drop(index=server_id_action['dismiss'], inplace=False)
    fleet = update_check_lifespan(fleet)
    return fleet


def put_fleet_on_hold(fleet):
    fleet['action'] = 'hold'
    fleet['moved'] = 0
    return fleet


def update_check_lifespan(fleet):
    # INCREASE LIFESPAN COUNTER AND DROP SERVERS THAT HAVE ACHIEVED THEIR
    # LIFE EXPECTANCY
    fleet['lifespan'] = fleet['lifespan'].fillna(0)
    fleet['lifespan'] += 1
    fleet = fleet.drop(fleet.index[fleet['lifespan'] > fleet['life_expectancy']], inplace=False)
    return fleet


def get_evaluation(fleet, 
                   pricing_strategy, 
                   demand,
                   datacenters,
                   servers,
                   selling_prices,
                   elasticity,
                   time_steps=get_known('time_steps'), 
                   verbose=1):

    # SOLUTION EVALUATION

    # SOLUTION DATA PREPARATION
    fleet = fleet_data_preparation(fleet, 
                                   servers, 
                                   datacenters, 
                                   selling_prices)

    # PRICING STRATEGY DATA PREPARATION
    pricing_strategy = pricing_data_preparation(pricing_strategy)
    elasticity = change_elasticity_format(elasticity)
    selling_prices = change_selling_prices_format(selling_prices)
    base_prices = selling_prices.copy()

    # DEMAND DATA PREPARATION
    demand = get_actual_demand(demand)
    OBJECTIVE = 0
    FLEET = pd.DataFrame()
    metrics = {
            'P': [],
            'R': [],
            'C': [],
            'D': {
                f'{sg}_{ls}': [] for sg in get_known('server_generation') for ls in get_known('latency_sensitivity')
            },
            'Z': {
                f'{sg}_{ls}': [] for sg in get_known('server_generation') for ls in get_known('latency_sensitivity')
            },
            'SLOTS': {
                'DC1': [],
                'DC2': [],
                'DC3': [],
                'DC4': [],
            }
        }
    # if ts-related fleet is empty then current fleet is ts-fleet
    for ts in range(1, time_steps+1):
        
        # GET THE ACTUAL DEMAND AT TIMESTEP ts
        D = get_time_step_demand(demand, ts)

        # GET THE SERVERS DEPLOYED AT TIMESTEP ts
        ts_fleet = get_time_step_fleet(fleet, ts)

        # GET THE PRICES AT TIMESTEP ts
        ts_prices = get_time_step_prices(pricing_strategy, ts)

        # UPDATE THE SELLING PRICES ACCORDING TO PRICES AT TIMESTEP ts
        selling_prices = update_selling_prices(selling_prices, ts_prices)

        # UPDATE THE DEMAND ACCORDING TO PRICES AT TIMESTEP ts
        D = update_demand_according_to_prices(D, selling_prices, base_prices, elasticity)

        if ts_fleet.empty and not FLEET.empty:
            ts_fleet = FLEET
        elif ts_fleet.empty and FLEET.empty:
            continue

        # UPDATE FLEET
        FLEET = update_fleet(ts, FLEET, ts_fleet)

        # CHECK IF THE FLEET IS EMPTY
        if FLEET.shape[0] > 0:
            # GET THE SERVERS CAPACITY AT TIMESTEP ts
            Zf = get_capacity_by_server_generation_latency_sensitivity(FLEET)

            # CHECK CONSTRAINTS
            slots = check_datacenter_slots_size_constraint(FLEET)
            for dc in get_known('datacenter_id'):
                if dc in slots.index:
                    metrics['SLOTS'][dc].append(float(slots.loc[dc, 'slots_size'] / slots.loc[dc, 'slots_capacity']))
                else:
                    metrics['SLOTS'][dc].append(0)
            # EVALUATE THE OBJECTIVE FUNCTION AT TIMESTEP ts
            # U = get_utilization(D, Zf)

            # L = get_normalized_lifespan(FLEET)

            P, R, C = get_profit(D, 
                           Zf, 
                           selling_prices,
                           FLEET)

            OBJECTIVE += P
            
            for sg in get_known('server_generation'):
                for ls in get_known('latency_sensitivity'):
                    if ls in Zf.columns:
                        # metrics['Z'][(sg, ls)].append(Zf[ls].get(sg, default=0))
                        metrics['Z'][f'{sg}_{ls}'].append(float(Zf[ls].get(sg, default=0)))
                    else:
                        # metrics['Z'][(sg, ls)].append(0)
                        metrics['Z'][f'{sg}_{ls}'].append(0)
                    metrics['D'][f'{sg}_{ls}'].append(float(D[ls].get(sg, default=0)))
            
            metrics['P'].append(float(P))
            metrics['R'].append(float(R))
            metrics['C'].append(float(C))
            # PUT ENTIRE FLEET on HOLD ACTION
            FLEET = put_fleet_on_hold(FLEET)

            # PREPARE OUTPUT
            output = {'time-step': ts,
                      'O': round(OBJECTIVE, 2),
                      'P': round(P, 2)}
        else:
            # PREPARE OUTPUT
            output = {'time-step': ts,
                      'O': np.nan,
                      'P': np.nan}

        if verbose:
            print(output)
            
    
    import rich
    rich.print('Debugging')
    rich.print(f"Profit : {sum(metrics['P']):0,.2f}")
    rich.print(f"- Revenue : {sum(metrics['R']):0,.2f}")
    rich.print(f"- Cost : {sum(metrics['C']):0,.2f}")
    
    return OBJECTIVE, metrics


def evaluation_function(fleet, 
                        pricing_strategy, 
                        demand,
                        datacenters,
                        servers,
                        selling_prices,
                        elasticity,
                        time_steps=get_known('time_steps'), 
                        seed=None,
                        verbose=0):

    """
    Evaluate a solution for the Tech Arena Phase 1 problem.

    Parameters
    ----------
    fleet : pandas DataFrame
        This is a fleet of servers. This is provided by the partecipant.
    pricing_strategy : pandas DataFrame
        This is a pricing strategy. This is provided by the partecipant.
    demand : pandas DataFrame
        This is the demand data. This is provided by default in the data 
        folder.
    datacenters : pandas DataFrame
        This is the datacenters data. This is provided by default in the data 
        folder.
    servers : pandas DataFrame
        This is the servers data. This is provided by default in the data 
        folder.
    selling_prices : pandas DataFrame
        This is the selling prices data. This is provided by default in the 
        data folder.
    elasticity : pandas DataFrame
        This is the price elasticity of demand data. This is provided by 
        default in the data folder.
    time_steps : int
        This is the number of time-steps for which we need to evaluate the 
        solution.
    c1_max_violations : int
        This is the maximum number of violations to Contraint 1 that can be
        tolerated. If this number is exceeded the function will output None.

    Return
    ------
    This function returns a float that represents the value of the objective
    function O evaluated across all time-steps.
    In case the solution cannot be evaluated the function returns None.
    """
    # SET RANDOM SEED
    np.random.seed(seed)
    # EVALUATE SOLUTION
    try:
        return get_evaluation(fleet, 
                                pricing_strategy, 
                                demand,
                                datacenters,
                                servers,
                                selling_prices,
                                elasticity,
                                time_steps=time_steps, 
                                verbose=verbose)
    # CATCH EXCEPTIONS
    except Exception as e:
        logger.error(e)
        return None

