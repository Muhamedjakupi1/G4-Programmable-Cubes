#!/usr/bin/env python3
"""
Random Search Baseline Algorithm for the International Space Station (ISS) Configuration Problem
Programmable Cubes Challenge - GECCO 2024 Space Optimisation Competition (SpOC)

This module implements a stochastic optimization baseline using pure random search
for the ISS spacecraft assembly problem. The algorithm serves as a benchmark for
comparative analysis against more sophisticated metaheuristic optimization approaches.

The random search methodology generates candidate solutions through uniform random
sampling of the solution space, providing an unbiased baseline for algorithmic
performance evaluation in the context of 3D modular spacecraft assembly optimization.

Usage:
    python solver/baselines/iss/random_search.py

Dependencies:
    - numpy: Numerical computing and array operations
    - random: Pseudorandom number generation
    - tqdm: Progress monitoring and visualization
    - matplotlib: Graphical visualization of results (optional)
    - json: Data serialization for result storage
"""

import sys
import os
import numpy as np
import random
import json
import time
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Add the src directory and the repository root to the Python path
# ... (Lines 38-40 are adding paths, which is good)
repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))
# ...

from src.programmable_cubes_UDP import programmable_cubes_UDP
# Modify this line to pass the root directory

# Experimental Configuration Parameters
N_ITERATIONS = 1000  # Number of stochastic sampling iterations
LOG_INTERVAL = 100  # Progress reporting frequency
RANDOM_SEED = 42  # Seed for pseudorandom number generator (reproducibility)
MAX_CHROMOSOME_LENGTH = 100  # Upper bound on solution representation length


def generate_random_chromosome(num_cubes, max_length=100):
    """
    Generate a stochastic chromosome representation for the programmable cubes optimization problem.

    This function implements uniform random sampling to create candidate solutions
    within the discrete action space defined by cube identifiers and movement commands.
    Each chromosome represents a sequence of cube-movement pairs terminated by a
    sentinel value (-1).

    Parameters:
        num_cubes (int): Total number of programmable cubes in the problem instance
        max_length (int): Maximum number of cube-movement command pairs allowed

    Returns:
        np.ndarray: Randomly generated chromosome with terminal sentinel value
    """
    # Generate stochastic sequence length within feasible bounds
    length = random.randint(1, max_length)

    chromosome = []
    for _ in range(length):
        # Uniform random sampling of cube identifier from discrete space [0, num_cubes-1]
        cube_id = random.randint(0, num_cubes - 1)
        # Uniform random sampling of movement command from discrete action space [0, 5]
        move_command = random.randint(0, 5)

        chromosome.extend([cube_id, move_command])

    # Append terminal sentinel value as per problem specification
    chromosome.append(-1)

    return np.array(chromosome)


def evaluate_chromosome(udp, chromosome):
    """
    Evaluate the objective function value for a given chromosome representation.

    This function interfaces with the User Defined Problem (UDP) to compute
    the fitness score, which quantifies the solution quality in terms of
    structural similarity to the target ISS configuration.

    Parameters:
        udp: Programmable cubes UDP instance containing problem definition
        chromosome (list): Chromosome representation to be evaluated

    Returns:
        float: Objective function value (negative values indicate better solutions)
    """
    try:
        fitness_score = udp.fitness(chromosome)
        return fitness_score[0]  # UDP returns fitness as single-element list
    except Exception as e:
        print(f"Evaluation error encountered: {e}")
        return float('-inf')  # Return worst possible objective value


def count_moves(chromosome):
    """
    Quantify the number of movement operations encoded in a chromosome.

    Parameters:
        chromosome (np.ndarray): The chromosome representation

    Returns:
        int: Number of cube-movement command pairs
    """
    # Locate terminal sentinel value position
    end_pos = np.where(chromosome == -1)[0][0]
    # Calculate number of movements (each move consists of cube_id + command)
    return end_pos // 2
