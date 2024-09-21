from pathlib import Path
from baseline.evaluator import eval_func
import rich 
import os
seeds = [2381, 5351, 6047, 6829, 9221, 9859, 8053, 1097, 8677, 2521]
import dotenv
dotenv.load_dotenv()
dataroot = '../../data'
scores = []
for seed in seeds:
    solution_file = Path(f'{seed}.json')
    assert solution_file.exists(), f'{solution_file} does not exist'
for seed in seeds:  
    solution_file = Path(f'{seed}.json')
    visualization_dir = Path(f'./visualization/{seed}')
    verbose = False
    print(f'Calculating score for seed {seed}')
    scores.append(eval_func(dataroot, solution_file, visualization_dir, seed, verbose))

average_score = sum(scores) / len(scores)
rich.print(f'Average score: {average_score:0,.2f}')
rich.print(' '.join([f'{score}' for score in scores]))