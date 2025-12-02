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
repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))

from programmable_cubes_UDP import programmable_cubes_UDP


# Experimental Configuration Parameters
N_ITERATIONS = 1000      # Number of stochastic sampling iterations
LOG_INTERVAL = 100       # Progress reporting frequency
RANDOM_SEED = 42         # Seed for pseudorandom number generator (reproducibility)
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
        plt.axvline(np.mean(fitness_history), color='red', linestyle='--', label=f'Mean: {np.mean(fitness_history):.6f}')
        plt.axvline(np.max(fitness_history), color='green', linestyle='--', label=f'Best: {np.max(fitness_history):.6f}')
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


def random_search_iss():
    """
    Execute stochastic optimization using random search methodology for the ISS assembly problem.
    
    This function implements a pure random search algorithm as a baseline optimization
    approach. The method performs uniform random sampling of the solution space to
    establish benchmark performance metrics for comparative analysis against more
    sophisticated metaheuristic algorithms.
    
    The algorithm maintains detailed performance statistics throughout the optimization
    process and persists results for subsequent empirical analysis.
    
    Returns:
        tuple: (best_chromosome, best_fitness, best_moves, results_data)
            - best_chromosome: Optimal solution representation found
            - best_fitness: Corresponding objective function value
            - best_moves: Number of movement operations in best solution
            - results_data: Complete experimental results dictionary
    """
    print("=" * 80)
    print("STOCHASTIC OPTIMIZATION: RANDOM SEARCH BASELINE FOR ISS ASSEMBLY")
    print("=" * 80)
    
    # Initialize pseudorandom number generator for reproducible experiments
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Initialize optimization problem instance
    print("Initializing User Defined Problem (UDP) for ISS configuration...")
    udp = programmable_cubes_UDP('ISS')
    
    # Extract problem parameters for algorithm configuration
    num_cubes = udp.setup['num_cubes']
    max_cmds = udp.setup['max_cmds']
    
    print(f"Problem Instance Characteristics:")
    print(f"  • Number of programmable cubes: {num_cubes}")
    print(f"  • Maximum movement commands: {max_cmds}")
    print(f"  • Optimization iterations: {N_ITERATIONS}")
    print(f"  • Random seed (reproducibility): {RANDOM_SEED}")
    print()
    
    # Initialize performance tracking variables
    best_fitness = float('inf')
    best_chromosome = None
    best_moves = 0
    fitness_history = []
    iteration_best_fitness = []
    
    # Record experimental metadata
    start_time = time.time()
    
    print("Commencing stochastic optimization process...")
    print()
    
    # Execute stochastic optimization iterations
    for iteration in tqdm(range(N_ITERATIONS), desc="Random Search Optimization"):
        # Generate candidate solution via uniform random sampling
        chromosome = generate_random_chromosome(num_cubes, max_length=min(200, max_cmds))
        
        # Evaluate objective function
        fitness = evaluate_chromosome(udp, chromosome)
        fitness_history.append(fitness)
        
        # Update global best solution if improvement found
        if fitness < best_fitness:
            best_fitness = fitness
            best_chromosome = chromosome.copy()
            best_moves = count_moves(chromosome)
            
            if (iteration + 1) % LOG_INTERVAL == 0:
                print(f"Iteration {iteration + 1:4d}: New optimum found | Fitness = {best_fitness:.6f} | Moves = {best_moves}")
        
        # Record current best for convergence analysis
        iteration_best_fitness.append(best_fitness)
        
        # Periodic progress reporting
        if (iteration + 1) % LOG_INTERVAL == 0 and fitness <= best_fitness:
            print(f"Iteration {iteration + 1:4d}: Current optimum | Fitness = {best_fitness:.6f} | Moves = {best_moves}")
    
    
    # Calculate optimization runtime
    execution_time = time.time() - start_time
    
    print()
    print("=" * 80)
    print("EXPERIMENTAL RESULTS AND STATISTICAL ANALYSIS")
    print("=" * 80)
    
    # Compute performance metrics
    print(f"Optimization Statistics:")
    print(f"  • Optimal objective function value: {best_fitness:.6f}")
    print(f"  • Number of movement operations: {best_moves}")
    print(f"  • Solution representation length: {len(best_chromosome)}")
    print(f"  • Computational runtime: {execution_time:.2f} seconds")
    print(f"  • Iterations per second: {N_ITERATIONS/execution_time:.1f}")
    
    # Calculate resource utilization metrics
    efficiency = (1 - best_moves / max_cmds) * 100
    print(f"  • Resource utilization efficiency: {efficiency:.1f}% ({best_moves}/{max_cmds} commands)")
    
    # Statistical analysis of fitness distribution
    fitness_array = np.array(fitness_history)
    print(f"\nFitness Distribution Analysis:")
    print(f"  • Mean fitness: {np.mean(fitness_array):.6f}")
    print(f"  • Standard deviation: {np.std(fitness_array):.6f}")
    print(f"  • Minimum fitness: {np.min(fitness_array):.6f}")
    print(f"  • Maximum fitness: {np.max(fitness_array):.6f}")
    
    print(f"\nOptimal Solution Representation (first 20 elements):")
    print(best_chromosome[:20], "..." if len(best_chromosome) > 20 else "")
    
    # Prepare comprehensive results data for comparative analysis
    results_data = {
        "algorithm": "Random Search",
        "problem": "ISS",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "n_iterations": N_ITERATIONS,
            "random_seed": RANDOM_SEED,
            "max_chromosome_length": MAX_CHROMOSOME_LENGTH
        },
        "problem_instance": {
            "num_cubes": num_cubes,
            "max_commands": max_cmds
        },
        "results": {
            "best_fitness": float(best_fitness),
            "best_moves": int(best_moves),
            "chromosome_length": int(len(best_chromosome)),
            "execution_time": float(execution_time),
            "iterations_per_second": float(N_ITERATIONS/execution_time),
            "resource_efficiency": float(efficiency)
        },
        "statistics": {
            "mean_fitness": float(np.mean(fitness_array)),
            "std_fitness": float(np.std(fitness_array)),
            "min_fitness": float(np.min(fitness_array)),
            "max_fitness": float(np.max(fitness_array))
        },
        "convergence_data": {
            "fitness_history": [float(f) for f in fitness_history],
            "best_fitness_evolution": [float(f) for f in iteration_best_fitness]
        },
        "solution": {
            "best_chromosome": [int(x) for x in best_chromosome]
        }
    }
    
    # Save experimental results for comparative analysis
    repo_root_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    results_dir = os.path.join(repo_root_path, 'solver', 'results')
    results_file = save_experimental_results(results_data, results_dir)
    
    # Generate and save comprehensive visualizations
    print(f"\nVisualization and Documentation:")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save solution visualizations (achieved vs target)
    saved_plots = save_solution_visualizations(udp, best_chromosome, results_dir, timestamp)
    
    # Save convergence analysis plot
    convergence_plot = save_convergence_plot(fitness_history, iteration_best_fitness, results_dir, timestamp)
    
    # Update results data with visualization paths
    visualization_paths = {
        "results_file": results_file,
        "plots": saved_plots
    }
    if convergence_plot:
        visualization_paths["convergence_plot"] = convergence_plot
    
    # Add visualization paths to results data and re-save
    results_data["visualization_files"] = visualization_paths
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"  • All experimental artifacts saved to: {results_dir}")
    print(f"  • Results file: {os.path.basename(results_file)}")
    if saved_plots:
        for plot_type, path in saved_plots.items():
            print(f"  • {plot_type.capitalize()} plot: {os.path.basename(path)}")
    if convergence_plot:
        print(f"  • Convergence analysis: {os.path.basename(convergence_plot)}")
    
    print()
    print("=" * 80)
    print("STOCHASTIC OPTIMIZATION COMPLETED SUCCESSFULLY")
    print("Results and visualizations saved for comparative algorithmic analysis")
    print("=" * 80)
    
    return best_chromosome, best_fitness, best_moves, results_data


if __name__ == "__main__":
    # Execute stochastic optimization experiment
    best_chromosome, best_fitness, best_moves, results_data = random_search_iss() 
