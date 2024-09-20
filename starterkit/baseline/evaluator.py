import argparse
from baseline.evaluation import evaluation_function, get_known
from pathlib import Path 
import os 
from dotenv import load_dotenv
import json
from baseline.utils import (load_problem_data,
                   load_solution)
import time

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f'{method.__name__} took: {te-ts:.2f} sec')
        return result
    return timed

@timeit
def run_evalution_function(**kwargs):
    return evaluation_function(**kwargs)

# Load environment variables
load_dotenv()
def eval_func(dataroot, solution_file, visualization_dir, seed, verbose=False):
    visualization_dir.mkdir(exist_ok=True, parents=True)
    solution_file = Path(solution_file)
    # Load the solution from the specified file
    if solution_file is None or not solution_file.exists():
        print(f'File {solution_file} not found.')
        print('Using the default solution file instead.')
        solution_file = Path(dataroot / 'solution_example.json')

    fleet, pricing_strategy = load_solution(solution_file)

    # Load problem data
    demand, datacenters, servers, selling_prices, elasticity = load_problem_data(path=dataroot)
    # Evaluate the solution
   

    score, metrics = run_evalution_function(fleet=fleet,
                                demand=demand,
                                pricing_strategy=pricing_strategy,
                                datacenters=datacenters,
                                servers=servers,
                                selling_prices=selling_prices,
                                elasticity=elasticity,
                                seed=int(seed), 
                                verbose=verbose)

    with open(visualization_dir / 'results.json', 'w') as f:
        json.dump(metrics, f)
    import matplotlib.pyplot as plt
    X = list(range(len(metrics['P'])))
    for metric in ['P']:
        values = metrics[metric]
        plt.plot(X, values, label=metric)
        plt.xlabel('Time')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(visualization_dir / f'{metric}.png')
        plt.clf()


    for sg in get_known('server_generation'):
        for ls in get_known('latency_sensitivity'):
            valuesD = metrics['D'][f'{sg}_{ls}']
            valuesZ = metrics['Z'][f'{sg}_{ls}']
            plt.plot(X, valuesD, label='Demand')
            plt.plot(X, valuesZ, label='Capacity')
            plt.xlabel('Time')
            plt.ylabel('computing unit')
            plt.legend()
            img_dir = visualization_dir / f'{sg}'
            img_dir.mkdir(exist_ok=True)
            plt.savefig(img_dir / f'ZD_{ls}.png')
            plt.clf()
    for dc in get_known('datacenter_id'):
        valuesC = metrics['SLOTS'][dc]
        plt.plot(X, valuesC, label='Using slots')
        plt.xlabel('Time')
        plt.ylabel('Slots Ratio')
        plt.legend()
        plt.savefig(visualization_dir / f'{dc}.png')
        plt.clf()
    return score
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='Path to the solution file', required=False)
    parser.add_argument('-s', '--seed', help='Random seed', required=False, default=123)
    parser.add_argument('-v', '--verbose', help='Print the solution score', action='store_true')
    parser.add_argument('-o', '--output', help='Output file', required=False, default='./output')
    args = parser.parse_args()


    solution_file = Path(args.file) if args.file else None
    # Set the data root directory
    dataroot = Path(os.getenv('DATAROOT'))
    assert dataroot.exists(), f"DATAROOT must be a valid path to the data directory, please set it in .env file\n Current value: {dataroot}"

    visualization_dir = Path(args.output)
    verbose = args.verbose
    seed = args.seed
    score = eval_func(dataroot, args.file, visualization_dir, seed, verbose)
    print(f'Score: {score}')
if __name__ == '__main__':
    main()