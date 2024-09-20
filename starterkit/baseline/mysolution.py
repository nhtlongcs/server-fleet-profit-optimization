
import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
from evaluation import get_actual_demand
from dotenv import load_dotenv
import os 
from pathlib import Path
load_dotenv()

def get_my_solution(d):
    # This is just a placeholder.
    return [{}]

dataroot = Path(os.getenv('DATAROOT'))
assert dataroot.exists(), "DATAROOT must be a valid path to the data directory"
seeds = known_seeds('training')

output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)

demand = pd.read_csv(dataroot / 'demand.csv')
for seed in seeds:
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)

    # CALL YOUR APPROACH HERE
    solution = get_my_solution(actual_demand)

    # SAVE YOUR SOLUTION
    save_solution(solution, f'./output/{seed}.json')

