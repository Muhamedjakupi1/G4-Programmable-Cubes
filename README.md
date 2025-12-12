# Programmable-Cubes

## Problem Description
This project tackles the optimization of assembling a space structure using programmable cubes. The task is to transform an initial cube configuration into a target final shape, using only allowed rotational moves, while respecting space constraints and a maximum command limit.

- Each cube can rotate around neighboring cubes.
- The goal is to find the most effective sequence of moves that transforms the structure as close as possible to the target configuration.
- The difference between the obtained final structure and the target is measured and minimized.

## Problem Instances
Three configurations of different sizes are analyzed:

- **ISS (International Space Station)** — 148 cubes, command limit 6000
- **JWST (James Webb Space Telescope)** — 643 cubes, command limit 30000
- **Enterprise** — 1472 cubes, command limit 100000

## Solution Approaches
Three main approaches are implemented:

### 1) Baselines
- **Random Search**: tries random solutions to provide a comparison baseline.

### 2) Heuristics
- **Greedy**: chooses the best move at the moment, but may get stuck in local optima.

### 3) Optimizers
- **Genetic Algorithm (GA)**: an evolutionary approach that improves a population of solutions through selection, crossover, and mutation.
- **Enhanced Genetic Algorithm**: an advanced version that runs multiple populations in parallel, preserves diversity, keeps memory of good solutions, and uses migration strategies to avoid local optima.

## Project Structure
```text
├── data/spoc3/cubes/          # Cube data and configurations (ISS, JWST, Enterprise)
├── problems/                  # Problem specification files
├── solver/                    # Optimization algorithms
│   ├── baselines/             # Random search implementation
│   ├── heuristics/            # Greedy algorithm implementation 
│   ├── optimizers/            # Genetic algorithm implementations
│   └── results/               # Results and visualizations
├── src/                       # Core components
│   ├── CubeMoveset.py         # Cube movement definitions
│   ├── programmable_cubes_UDP.py  # PyGMO problem interfac
│   └── submission_helper.py   # Competition submission utilities
└── submissions/               # Generated solution files
```


## Programming Language and Libraries
- **Python**
- **NumPy** — numerical computations
- **Numba** — performance optimization
- **Matplotlib** — visualizations
- **PyGMO** — optimization algorithms

## Running the Algorithms
Example run scenarios:

```bash
python solver/optimizers/iss/submit_ga_iss.py
python solver/optimizers/enterprise/submit_enhanced_ga_enterprise.py
python solver/heuristics/jwst/submit_greedy_jwst.py
```


