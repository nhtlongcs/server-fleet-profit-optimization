

import json
import pandas as pd
from os.path import abspath, join


def load_json(path):
    return json.load(open(path, encoding='utf-8'))


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as out:
        json.dump(data, out, ensure_ascii=False, indent=4)


def load_solution(path):
    # Loads a solution from a json file to 2 pandas DataFrames.
    solution = load_json(path)
    fleet = pd.DataFrame(solution['fleet'])
    pricing_strategy = pd.DataFrame(solution['pricing_strategy'])
    return fleet, pricing_strategy


def save_solution(fleet, pricing_strategy, path):
    # Saves a solution into a json file.
    fleet = fleet.to_dict('records')
    pricing_strategy = pricing_strategy.to_dict('records')
    solution = {'fleet': fleet,
                'pricing_strategy': pricing_strategy}
    return save_json(path, solution)


def load_problem_data(path=None):
    if path is None:
        path = './data/'

    # LOAD DEMAND
    p = abspath(join(path, 'demand.csv'))
    demand = pd.read_csv(p)    
    
    # LOAD DATACENTERS DATA
    p = abspath(join(path, 'datacenters.csv'))
    datacenters = pd.read_csv(p)
    
    # LOAD SERVERS DATA
    p = abspath(join(path, 'servers.csv'))
    servers = pd.read_csv(p)
    
    # LOAD SELLING PRICES DATA
    p = abspath(join(path, 'selling_prices.csv'))
    selling_prices = pd.read_csv(p)

    # LOAD ELASTICITY DATA
    p = abspath(join(path, 'price_elasticity_of_demand.csv'))
    elasticity = pd.read_csv(p)
    return demand, datacenters, servers, selling_prices, elasticity


if __name__ == '__main__':


    # Load solution
    path = './data/solution_example.json'

    fleet, pricing_strategy = load_solution(path)

    print(fleet)
    print(pricing_strategy)




