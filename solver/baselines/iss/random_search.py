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

def save_experimental_results(results_data, output_dir):
    """
    Persist experimental results for subsequent comparative analysis.

    This function serializes optimization results to JSON format, enabling
    systematic comparison between different algorithmic approaches and
    statistical analysis of performance metrics.

    Parameters:
        results_data (dict): Dictionary containing experimental results and metadata
        output_dir (str): Directory path for result storage

    Returns:
        str: Path to the saved results file
    """
    # Ensure results directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for unique file identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"random_search_iss_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Serialize results with proper formatting
    with open(filepath, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"Experimental results saved to: {filepath}")
    return filepath


def save_solution_visualizations(udp, best_chromosome, output_dir, timestamp):
    """
    Generate and save visualization plots of the optimal solution.

    This function creates visual representations of both the achieved configuration
    and the target configuration, saving them as high-quality images for
    documentation and analysis purposes.

    Parameters:
        udp: Programmable cubes UDP instance
        best_chromosome: Optimal solution representation
        output_dir (str): Directory path for saving plots
        timestamp (str): Timestamp for unique file naming

    Returns:
        dict: Paths to saved visualization files
    """
    # Ensure results directory exists
    os.makedirs(output_dir, exist_ok=True)

    saved_plots = {}

    try:
        # Evaluate optimal solution to configure UDP state
        udp.fitness(best_chromosome)

        # Save ensemble (achieved) configuration
        print("  • Generating and saving ensemble configuration visualization...")
        plt.figure(figsize=(12, 8))
        udp.plot('ensemble')
        ensemble_path = os.path.join(output_dir, f"random_search_iss_ensemble_{timestamp}.png")
        plt.savefig(ensemble_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        saved_plots['ensemble'] = ensemble_path
        print(f"    Ensemble plot saved: {ensemble_path}")

        # Save target configuration
        print("  • Generating and saving target configuration visualization...")
        plt.figure(figsize=(12, 8))
        udp.plot('target')
        target_path = os.path.join(output_dir, f"random_search_iss_target_{timestamp}.png")
        plt.savefig(target_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        saved_plots['target'] = target_path
        print(f"    Target plot saved: {target_path}")

        # Generate convergence plot
        print("  • Generating and saving convergence analysis...")

    except Exception as e:
        print(f"  • Visualization error: {e}")
        print("  • Note: Some visualizations may require specific dependencies")

    return saved_plots


def save_convergence_plot(fitness_history, best_fitness_evolution, output_dir, timestamp):
    """
    Generate and save convergence analysis plot.

    This function creates a visualization showing the optimization progress
    over iterations, including both the fitness history and best fitness evolution.

    Parameters:
        fitness_history (list): List of fitness values for each iteration
        best_fitness_evolution (list): List of best fitness values over iterations
        output_dir (str): Directory path for saving plots
        timestamp (str): Timestamp for unique file naming

    Returns:
        str: Path to saved convergence plot
    """
    try:
        plt.figure(figsize=(14, 6))

        # Create subplot for fitness history
        plt.subplot(1, 2, 1)
        plt.plot(fitness_history, alpha=0.6, color='lightblue', label='Individual Evaluations')
        plt.plot(best_fitness_evolution, color='darkblue', linewidth=2, label='Best Fitness Evolution')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value')
        plt.title('Random Search Convergence Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Create subplot for fitness distribution
        plt.subplot(1, 2, 2)
        plt.hist(fitness_history, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(fitness_history), color='red', linestyle='--',
                    label=f'Mean: {np.mean(fitness_history):.6f}')
        plt.axvline(np.max(fitness_history), color='green', linestyle='--',
                    label=f'Best: {np.max(fitness_history):.6f}')
        plt.xlabel('Fitness Value')
        plt.ylabel('Frequency')
        plt.title('Fitness Distribution Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        convergence_path = os.path.join(output_dir, f"random_search_iss_convergence_{timestamp}.png")
        plt.savefig(convergence_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"    Convergence plot saved: {convergence_path}")
        return convergence_path

    except Exception as e:
        print(f"  • Convergence plot error: {e}")
        return None
