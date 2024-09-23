## Tech Arena Baseline Description

<!-- huawei-server-fleet-management -->

### Problem Statement

The problem is to manage a fleet of servers in a data center. The data center has four locations (DC1, DC2, DC3, DC4) and each location has a fleet of servers. The servers are of two types: CPU and GPU. Each server has a unique id, and a generation (S1, S2, S3, S4). The servers are managed in time steps. At each time step, the data center manager can take one of the following actions: buy, hold, move, and dismiss. The goal is to maximize the objective function of each data center by managing the servers efficiently.

See the [problem statement](docs/problem_statement.pdf) pdf for more details.

### Environment Setup

Install Conda by following the instructions [here](https://github.com/conda-forge/miniforge) 

For example, on Linux x86_64 (amd64), you can install Miniforge with the following command:

```bash

wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge-pypy3-Linux-x86_64.sh
sh Miniforge-pypy3-Linux-x86_64.sh
# And follow the instructions on the screen

```

Install the required packages by running the following commands:

```bash
cd starterkit/
mamba env create -f env.yml
conda activate fleet
```

### Submission

To reproduce the baseline solution, run the following command:

```bash
cd experiments/buy_move_dismiss/
sh gen.sh
```

### Evaluation

To evaluate your solution, you need to provide a json file with the following format:

```json

{
    "fleet": [
        {
            "time_step": ,
            "datacenter_id": <DC1-DC4>,
            "server_generation": <CPU.S1-4 | GPU.S1-4>,
            "server_id": <server_id>,
            "action": <buy|hold|required|dissmiss>
        },
        ...
    ], 
    "pricing_strategy": {
        "time_step": ,
        "server_generation": <CPU.S1-4 | GPU.S1-4>,
        "latency_sensitive": <low|med|high>,
        "pricing": <pricing>
    }
}
// [{"time_step": 1, "datacenter_id": "DC1", "server_generation": "CPU.S1", "server_id": "7ee8a1de-b4b8-4fce-9bd6-19fdf8c1e409", "action": "buy"}]
```

The evaluation script is provided in the `baseline/` directory. To evaluate your solution, run the following command:

```bash
fleet-eval -f <file_path> -s <seed> 
```

