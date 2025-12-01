#!/usr/bin/env python3
"""
Enhanced Genetic Algorithm for ISS Spacecraft Assembly Problem
Programmable Cubes Challenge - GECCO 2024 Space Optimisation Competition (SpOC)

This module implements an advanced genetic algorithm with comprehensive experimental
documentation, visualization capabilities, and result analysis for the International
Space Station spacecraft assembly optimization problem. The algorithm features
adaptive mechanisms, intelligent initialization strategies, and systematic
performance monitoring for competitive optimization results.

The genetic algorithm employs multi-strategy population initialization, tournament
selection, adaptive crossover and mutation operators, elite preservation, and
local search enhancement. Fitness direction optimization ensures proper convergence
toward negative fitness values, indicating superior assembly configurations.

Key algorithmic enhancements include:
- Corrected fitness direction optimization (negative values indicate better solutions)
- Inverse-move cleanup for chromosome efficiency optimization
- Adaptive mutation rate mechanisms responding to optimization stagnation
- Comprehensive experimental data collection and academic visualization

Target Performance: Achieve fitness of -0.991 or superior (championship-level performance)

Usage:
    python solver/optimizers/iss/ga_solver.py

Dependencies:
    - numpy: Numerical computing and array operations
    - matplotlib: Result visualization and plotting
    - tqdm: Progress monitoring during optimization
    - scipy: Distance calculations and scientific computing
    - json: Experimental data serialization
"""

import sys
import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for automated plot generation
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime
from tqdm import tqdm
from scipy.spatial.distance import cdist
from collections import defaultdict

# Add the src directory and the repository root to the Python path
repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))
from programmable_cubes_UDP import programmable_cubes_UDP

import sys
import os
import numpy as np
import random
import json
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for automated plot generation
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from scipy.spatial.distance import cdist
from collections import defaultdict
import copy

# Add the src directory and the repository root to the Python path
repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))

from programmable_cubes_UDP import programmable_cubes_UDP

# Advanced Genetic Algorithm Configuration Parameters
POPULATION_SIZE = 100               # Population size for evolutionary optimization
GENERATIONS = 250                   # Maximum number of evolutionary generations
TOURNAMENT_SIZE = 5                 # Tournament selection pressure parameter
ELITE_SIZE = 10                    # Elitism preservation count for best individuals
CROSSOVER_RATE = 0.85              # Crossover probability for genetic recombination
BASE_MUTATION_RATE = 0.08          # Base mutation rate for genetic diversity
MAX_MUTATION_RATE = 0.25           # Maximum adaptive mutation rate
MAX_CHROMOSOME_LENGTH = 800        # Maximum chromosome length constraint
MIN_CHROMOSOME_LENGTH = 80         # Minimum chromosome length requirement
RANDOM_SEED = 42                   # Deterministic seed for reproducible experiments
LOG_INTERVAL = 20                  # Progress reporting frequency

# Initialization Strategy Distribution
SMART_INITIALIZATION_RATIO = 0.5   # Intelligent initialization proportion
GREEDY_INITIALIZATION_RATIO = 0.3  # Greedy initialization proportion  
RANDOM_RATIO = 0.2                 # Random initialization proportion

# Local Search and Optimization Enhancement Parameters
LOCAL_SEARCH_RATE = 0.3            # Local search application probability
LOCAL_SEARCH_ITERATIONS = 15       # Local search iteration depth
CLEANUP_RATE = 0.8                 # Inverse-move cleanup application rate

# Adaptive Algorithm Parameters
STAGNATION_THRESHOLD = 25          # Stagnation detection threshold
ADAPTIVE_MUTATION_FACTOR = 1.5     # Mutation rate amplification factor

# Academic Results and Documentation Configuration
RESULTS_DIR = "solver/results/iss/optimizers"  # Academic output directory for experimental data

def save_experimental_results(results_data):
    """
    Persist comprehensive experimental results to academic-standard JSON format.
    
    Args:
        results_data (dict): Complete experimental results and metadata
        
    Returns:
        str: Path to saved results file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(repo_root, RESULTS_DIR)
    os.makedirs(results_path, exist_ok=True)
    
    filename = f"genetic_algorithm_iss_experiment_{timestamp}.json"
    filepath = os.path.join(results_path, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"Experimental results saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving experimental results: {e}")
        return None

def save_solution_visualizations(udp, best_chromosome, output_dir, timestamp):
    """
    Generate and save visualizations of the genetic algorithm optimization results.
    
    This function creates comprehensive plots showing both the achieved ensemble
    configuration and the target configuration for comparative analysis of the
    genetic algorithm optimization performance.
    
    Args:
        udp: The programmable cubes UDP instance
        best_chromosome: The optimal solution chromosome found by genetic algorithm
        output_dir (str): Directory path for saving plots
        timestamp (str): Timestamp for unique file naming
        
    Returns:
        dict: Dictionary containing paths to saved plots
    """
    saved_plots = {}
    
    try:
        # Evaluate the best solution to set final cube positions
        print(f"  • Evaluating best solution for visualization...")
        udp.fitness(best_chromosome)
        
        # Save ensemble (achieved) configuration
        print("  • Generating and saving ensemble configuration visualization...")
        plt.figure(figsize=(12, 8))
        udp.plot('ensemble')
        ensemble_path = os.path.join(output_dir, f"genetic_algorithm_iss_ensemble_{timestamp}.png")
        plt.savefig(ensemble_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        saved_plots['ensemble'] = ensemble_path
        print(f"    Ensemble plot saved: {ensemble_path}")
        
        # Save target configuration
        print("  • Generating and saving target configuration visualization...")
        plt.figure(figsize=(12, 8))
        udp.plot('target')
        target_path = os.path.join(output_dir, f"genetic_algorithm_iss_target_{timestamp}.png")
        plt.savefig(target_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        saved_plots['target'] = target_path
        print(f"    Target plot saved: {target_path}")
        
    except Exception as e:
        print(f"  • Visualization error: {e}")
        print(f"  • Error details: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"  • Traceback: {traceback.format_exc()}")
        print("  • Note: Some visualizations may require specific dependencies")
    
    return saved_plots

def save_convergence_plot(fitness_history, best_fitness_evolution, results_path, timestamp):
    """
    Generate and save convergence analysis plot with performance metrics.
    
    This function creates a comprehensive visualization showing the optimization
    progress over generations, including both the fitness evolution and distribution
    analysis for comprehensive performance assessment.
    
    Args:
        fitness_history (list): Complete fitness evolution history from all evaluations
        best_fitness_evolution (list): Best fitness progression over generations
        results_path (str): Directory path for results storage
        timestamp (str): Timestamp for file naming
        
    Returns:
        str: Path to saved convergence plot or None if error
    """
    try:
        plt.figure(figsize=(14, 6))
        
        # Create subplot for fitness evolution analysis
        plt.subplot(1, 2, 1)
        generations = range(1, len(best_fitness_evolution) + 1)
        
        # Plot individual fitness evaluations if available
        if len(fitness_history) > len(best_fitness_evolution):
            plt.plot(fitness_history, alpha=0.6, color='lightblue', label='Individual Evaluations', marker='.')
        
        # Plot best fitness evolution
        plt.plot(generations, best_fitness_evolution, color='darkblue', linewidth=2, label='Best Fitness Evolution', marker='o')
        
        # Add target fitness reference line
        plt.axhline(y=-0.991, color='red', linestyle='--', linewidth=2, label='Target Fitness (-0.991)')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.title('Genetic Algorithm Convergence Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Create subplot for fitness distribution analysis
        plt.subplot(1, 2, 2)
        all_fitness_values = fitness_history if len(fitness_history) > len(best_fitness_evolution) else best_fitness_evolution
        
        plt.hist(all_fitness_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(all_fitness_values), color='red', linestyle='--', label=f'Mean: {np.mean(all_fitness_values):.6f}')
        plt.axvline(np.min(all_fitness_values), color='green', linestyle='--', label=f'Best: {np.min(all_fitness_values):.6f}')
        plt.axvline(-0.991, color='orange', linestyle=':', label='Target: -0.991')
        
        plt.xlabel('Fitness Value')
        plt.ylabel('Frequency')
        plt.title('Fitness Distribution Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        convergence_path = os.path.join(results_path, f"genetic_algorithm_iss_convergence_{timestamp}.png")
        plt.savefig(convergence_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Convergence plot saved: {convergence_path}")
        return convergence_path
        
    except Exception as e:
        print(f"    Convergence plot error: {e}")
        return None
LOCAL_SEARCH_RATE = 0.3
LOCAL_SEARCH_ITERATIONS = 15
CLEANUP_RATE = 0.8  # Rate of applying inverse-move cleanup

# Adaptive parameters
STAGNATION_THRESHOLD = 25
ADAPTIVE_MUTATION_FACTOR = 1.5
