
import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
from evaluation import get_actual_demand


def get_my_solution(d):
    # This is just a placeholder.
    return pd.DataFrame(), pd.DataFrame()


seeds = known_seeds()

demand = pd.read_csv('./data/demand.csv')
for seed in seeds:
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)

    # CALL YOUR APPROACH HERE
    fleet, pricing_strategy = get_my_solution(actual_demand)

    # SAVE YOUR SOLUTION
    save_solution(fleet, pricing_strategy, f'./output/{seed}.json')

