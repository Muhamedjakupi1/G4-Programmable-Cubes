#!/usr/bin/env python3
"""
Stochastic Optimization Baseline for JWST Spacecraft Assembly Problem
Academic Implementation for GECCO 2024 Space Optimization Competition (SpOC)

This module implements a rigorous stochastic optimization baseline using Monte Carlo
random search methodology for the James Webb Space Telescope (JWST) reconfiguration
problem. The implementation serves as a fundamental benchmark for comparative analysis
of advanced metaheuristic optimization algorithms in the domain of modular spacecraft
assembly and autonomous cube reconfiguration.

The algorithm employs unbiased stochastic sampling of the solution space to establish
baseline performance metrics for the complex combinatorial optimization problem of
transforming initial cube ensemble configurations to target spatial arrangements.

Academic Usage:
    python solver/baselines/jwst/random_search.py

Research Dependencies:
    - numpy: Numerical computing and array operations
    - random: Stochastic number generation
    - tqdm: Progress monitoring for iterative processes
    - matplotlib: Scientific visualization and plotting
    - json: Structured data serialization
    - datetime: Experimental timestamp generation
"""

import sys
import os
import numpy as np
import random
import json
import datetime
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for automated plotting
import matplotlib.pyplot as plt

# Add the src directory and the repository root to the Python path
repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))

from src.programmable_cubes_UDP import programmable_cubes_UDP

# Research Configuration Parameters
N_ITERATIONS = 1000  # Experimental sample size for stochastic exploration
LOG_INTERVAL = 100  # Progress reporting frequency for convergence monitoring
RANDOM_SEED = 42  # Reproducibility seed for experimental validation
RESULTS_DIR = "solver/results/jwst"  # Academic output directory for experimental data


def generate_stochastic_chromosome(num_cubes, max_length=1500):
    """
    Generate a stochastic chromosome for the programmable cubes optimization problem.

    This function implements Monte Carlo sampling to create random solution vectors
    for the combinatorial optimization space of cube assembly sequences.

    Args:
        num_cubes (int): Total number of modular cubes in the problem instance
        max_length (int): Maximum number of cube-move pairs in the solution sequence

    Returns:
        np.ndarray: A stochastically generated chromosome terminated with -1 sentinel
    """
    # Generate random length for the chromosome using uniform distribution
    length = random.randint(1, max_length)

    chromosome = []
    for _ in range(length):
        # Stochastic selection of cube identifier (0 to num_cubes-1)
        cube_id = random.randint(0, num_cubes - 1)
        # Stochastic selection of movement command (0 to 5)
        move_command = random.randint(0, 5)

        chromosome.extend([cube_id, move_command])

    # Ensure chromosome terminates with sentinel value
    chromosome.append(-1)

    return np.array(chromosome)


def evaluate_fitness_function(udp, chromosome):
    """
    Evaluate the objective fitness function for a given solution chromosome.

    This function interfaces with the User Defined Problem (UDP) to compute
    the fitness score representing solution quality in the optimization landscape.

    Args:
        udp: The programmable cubes UDP instance for fitness evaluation
        chromosome (list): The solution chromosome to evaluate

    Returns:
        float: The fitness score (negative value indicates better performance)
    """
    try:
        fitness_score = udp.fitness(chromosome)
        return fitness_score[0]  # UDP returns a list with one element
    except Exception as e:
        print(f"Error in fitness evaluation: {e}")
        return float('-inf')  # Return worst possible fitness for invalid solutions


def quantify_solution_complexity(chromosome):
    """
    Quantify the complexity of a solution in terms of movement operations.

    This function analyzes the solution chromosome to determine the number
    of discrete movement operations required for ensemble reconfiguration.

    Args:
        chromosome (np.ndarray): The solution chromosome to analyze

    Returns:
        int: Number of movement operations (cube-command pairs)
    """
    # Locate the sentinel terminator position
    end_pos = np.where(chromosome == -1)[0][0]
    # Calculate number of operations (half the length due to cube_id + command pairs)
    return end_pos // 2


def save_experimental_results(best_chromosome, best_fitness, best_moves, execution_time, num_cubes, max_cmds):
    """
    Save comprehensive experimental results for academic analysis and comparative studies.

    This function creates structured data files containing detailed experimental
    outcomes for subsequent statistical analysis and algorithmic comparison.

    Args:
        best_chromosome: Optimal solution chromosome discovered
        best_fitness: Achieved fitness score
        best_moves: Number of movement operations
        execution_time: Algorithm execution duration
        num_cubes: Problem instance size (number of cubes)
        max_cmds: Maximum allowed commands
    """
    # Create results directory if it doesn't exist
    results_path = os.path.join(repo_root, RESULTS_DIR)
    os.makedirs(results_path, exist_ok=True)

    # Generate timestamp for experimental session
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Compile comprehensive experimental data
    experimental_data = {
        "algorithm": "Stochastic Random Search",
        "problem_instance": "JWST Spacecraft Assembly",
        "experimental_parameters": {
            "iterations": N_ITERATIONS,
            "random_seed": RANDOM_SEED,
            "problem_size": num_cubes,
            "max_commands": max_cmds
        },
        "performance_metrics": {
            "best_fitness": float(best_fitness),
            "solution_complexity": int(best_moves),
            "execution_time_seconds": float(execution_time),
            "convergence_efficiency": float(best_moves / max_cmds) if max_cmds > 0 else 0.0
        },
        "solution_data": {
            "chromosome_length": len(best_chromosome),
            "chromosome": best_chromosome.tolist() if isinstance(best_chromosome, np.ndarray) else best_chromosome
        },
        "metadata": {
            "timestamp": timestamp,
            "algorithm_type": "baseline_stochastic",
            "optimization_objective": "minimize_negative_fitness"
        }
    }

    # Save experimental results
    results_file = os.path.join(results_path, f"stochastic_search_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(experimental_data, f, indent=2)

    print(f"Experimental results saved: {results_file}")


def save_solution_visualizations(udp, best_chromosome, output_dir, timestamp):
    """
    Generate and save visualizations of the optimization results.

    This function creates plots showing both the achieved ensemble
    configuration and the target configuration for comparative analysis.

    Args:
        udp: The programmable cubes UDP instance
        best_chromosome: The optimal solution chromosome
        output_dir (str): Directory path for saving plots
        timestamp (str): Timestamp for unique file naming

    Returns:
        dict: Dictionary containing paths to saved plots
    """
    saved_plots = {}

    try:
        # Evaluate the best solution to set final cube positions
        udp.fitness(best_chromosome)

        # Save ensemble (achieved) configuration
        print("  • Generating and saving ensemble configuration visualization...")
        plt.figure(figsize=(12, 8))
        udp.plot('ensemble')
        ensemble_path = os.path.join(output_dir, f"random_search_jwst_ensemble_{timestamp}.png")
        plt.savefig(ensemble_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        saved_plots['ensemble'] = ensemble_path
        print(f"    Ensemble plot saved: {ensemble_path}")

        # Save target configuration
        print("  • Generating and saving target configuration visualization...")
        plt.figure(figsize=(12, 8))
        udp.plot('target')
        target_path = os.path.join(output_dir, f"random_search_jwst_target_{timestamp}.png")
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

        convergence_path = os.path.join(output_dir, f"random_search_jwst_convergence_{timestamp}.png")
        plt.savefig(convergence_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"    Convergence plot saved: {convergence_path}")
        return convergence_path

    except Exception as e:
        print(f"  • Convergence plot error: {e}")
        return None


def stochastic_search_jwst():
    """
    Main stochastic optimization algorithm for the JWST spacecraft assembly problem.

    This function implements a comprehensive Monte Carlo random search methodology
    for exploring the combinatorial optimization landscape of modular spacecraft
    assembly. The algorithm performs systematic stochastic sampling to establish
    baseline performance metrics for comparative algorithmic analysis.

    Returns:
        tuple: (best_chromosome, best_fitness, best_moves, fitness_history)
    """
    print("=" * 80)
    print("STOCHASTIC OPTIMIZATION BASELINE FOR JWST SPACECRAFT ASSEMBLY")
    print("Academic Implementation - GECCO 2024 Space Optimization Competition")
    print("=" * 80)

    # Initialize reproducible stochastic environment
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- FIX START: Change CWD to repo_root for correct relative path resolution ---
    # Temporarily change the Current Working Directory (CWD) to the repository root.
    # This resolves issues where internal functions (like UDP.__init__ and UDP.fitness)
    # use relative paths defined in JWST.json (e.g., "data/spoc3/...") which must
    # be relative to the project root, not the script's execution directory.
    original_cwd = os.getcwd()
    try:
        os.chdir(repo_root)

        # Initialize User Defined Problem for JWST configuration
        print("Initializing UDP for JWST spacecraft assembly problem...")
        udp = programmable_cubes_UDP('JWST')

        # Extract problem instance parameters
        num_cubes = udp.setup['num_cubes']
        max_cmds = udp.setup['max_cmds']

        print(f"Problem Instance Characteristics:")
        print(f"  • Number of modular cubes: {num_cubes}")
        print(f"  • Maximum command sequence length: {max_cmds}")
        print(f"  • Stochastic sample size: {N_ITERATIONS}")
        print(f"  • Reproducibility seed: {RANDOM_SEED}")
        print()

        # Initialize optimization tracking variables
        best_fitness = float('inf')
        best_chromosome = None
        best_moves = 0
        fitness_history = []
        iteration_best_fitness = []

        import time
        start_time = time.time()

        print("Initiating stochastic exploration of solution space...")
        print()

        # Main stochastic optimization loop
        for iteration in tqdm(range(N_ITERATIONS), desc="Stochastic Search Progress"):
            # Generate stochastic solution candidate
            max_chromosome_length = min(3000, max_cmds // 10)  # Scale appropriately for JWST
            chromosome = generate_stochastic_chromosome(num_cubes, max_length=max_chromosome_length)

            # Evaluate solution quality using fitness function
            fitness = evaluate_fitness_function(udp, chromosome)
            fitness_history.append(fitness)

            # Update optimal solution if improvement discovered
            if fitness < best_fitness:
                best_fitness = fitness
                best_chromosome = chromosome.copy()
                best_moves = quantify_solution_complexity(chromosome)

                if (iteration + 1) % LOG_INTERVAL == 0:
                    print(
                        f"Iteration {iteration + 1:4d}: Enhanced fitness = {best_fitness:.6f} (operations: {best_moves})")

            # Track best fitness evolution
            iteration_best_fitness.append(best_fitness)

            # Periodic progress reporting
            if (iteration + 1) % LOG_INTERVAL == 0 and fitness <= best_fitness:
                print(
                    f"Iteration {iteration + 1:4d}: Current optimal fitness = {best_fitness:.6f} (operations: {best_moves})")

        execution_time = time.time() - start_time

        print()
        print("=" * 80)
        print("EXPERIMENTAL RESULTS AND PERFORMANCE ANALYSIS")
        print("=" * 80)

        # Comprehensive results analysis
        print(f"Optimal fitness achieved: {best_fitness:.6f}")
        print(f"Solution complexity (operations): {best_moves}")
        print(f"Chromosome encoding length: {len(best_chromosome)}")
        print(f"Algorithm execution time: {execution_time:.2f} seconds")

        # Calculate efficiency metrics
        efficiency = (1 - best_moves / max_cmds) * 100 if max_cmds > 0 else 0
        print(f"Operational efficiency: {efficiency:.1f}% ({best_moves}/{max_cmds} operations utilized)")

        print()
        print("Optimal chromosome encoding (first 20 elements):")
        print(best_chromosome[:20], "..." if len(best_chromosome) > 20 else "")

        # Generate academic visualizations
        print()
        print("Generating scientific visualizations and experimental data...")

        # Save comprehensive experimental results
        save_experimental_results(best_chromosome, best_fitness, best_moves,
                                  execution_time, num_cubes, max_cmds)

        # Generate solution visualizations and convergence plots
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(repo_root, RESULTS_DIR)

            # Save configuration plots
            saved_plots = save_solution_visualizations(udp, best_chromosome, results_path, timestamp)

            # Save convergence analysis
            convergence_path = save_convergence_plot(fitness_history, iteration_best_fitness, results_path, timestamp)

        except Exception as e:
            print(f"Visualization generation error: {e}")
            print("Note: Visualization functionality may require display environment")

        print()
        print("Stochastic optimization analysis completed successfully!")
        print("Experimental data and visualizations saved for comparative research")

        return best_chromosome, best_fitness, best_moves, fitness_history

    finally:
        # Crucial step: Restore the original CWD regardless of execution success or failure
        os.chdir(original_cwd)
    # --- FIX END ---


if __name__ == "__main__":
    # Execute the stochastic optimization analysis
    best_chromosome, best_fitness, best_moves, fitness_history = stochastic_search_jwst()