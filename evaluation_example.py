

from utils import (load_problem_data,
                   load_solution)
from evaluation import evaluation_function


# LOAD SOLUTION
fleet, pricing_strategy = load_solution('./data/solution_example.json')

# LOAD PROBLEM DATA
demand, datacenters, servers, selling_prices, elasticity = load_problem_data()

# EVALUATE THE SOLUTION
score = evaluation_function(fleet, 
                            pricing_strategy,
                            demand,
                            datacenters,
                            servers,
                            selling_prices,
                            elasticity,
                            seed=123)

print(f'Solution score: {score}')



