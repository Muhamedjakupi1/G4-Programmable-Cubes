import sys
import os
import numpy as np
import random
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for automated plot generation
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime
from tqdm import tqdm
from scipy.spatial.distance import cdist
from collections import defaultdict, deque
import copy
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Add the src directory and the repository root to the Python path
repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))

from programmable_cubes_UDP import programmable_cubes_UDP

# Enhanced Configuration for Enterprise (scaled appropriately)
POPULATION_SIZE = 60  # Reduced for larger problem space
GENERATIONS = 150  # Reduced for computational efficiency
TOURNAMENT_SIZE = 5  # Maintained
ELITE_SIZE = 8  # Reduced for Enterprise scale
CROSSOVER_RATE = 0.9  # High crossover rate
BASE_MUTATION_RATE = 0.05  # Lower for larger problem
MAX_MUTATION_RATE = 0.25  # Reasonable maximum
MAX_CHROMOSOME_LENGTH = 2000  # Increased for Enterprise complexity
MIN_CHROMOSOME_LENGTH = 80  # Reasonable minimum

# Multi-population parameters (scaled for Enterprise)
NUM_POPULATIONS = 2  # Reduced for computational efficiency
MIGRATION_RATE = 0.08  # Slightly higher migration
MIGRATION_INTERVAL = 25  # Longer intervals

# Advanced parameters (Enterprise-optimized)
DIVERSITY_THRESHOLD = 0.03
NOVELTY_ARCHIVE_SIZE = 30  # Reduced for efficiency
MEMORY_SIZE = 60  # Reduced for efficiency
TABU_SIZE = 20  # Reduced
LOG_INTERVAL = 10  # More frequent logging

# Problem-specific parameters
SMART_INITIALIZATION_RATIO = 0.7  # More smart individuals
GREEDY_INITIALIZATION_RATIO = 0.25
RANDOM_RATIO = 0.05  # Less random for Enterprise

LOCAL_SEARCH_RATE = 0.3  # Moderate local search
LOCAL_SEARCH_ITERATIONS = 15  # Reasonable iterations
CLEANUP_RATE = 0.9  # High cleanup rate

STAGNATION_THRESHOLD = 15  # Faster adaptation for Enterprise
ADAPTIVE_MUTATION_FACTOR = 1.6

# Academic Results and Documentation Configuration
RESULTS_DIR = "solver/results/enterprise/optimizers"  # Academic output directory for experimental data


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

    filename = f"enhanced_genetic_algorithm_enterprise_experiment_{timestamp}.json"
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
        ensemble_path = os.path.join(output_dir, f"enhanced_genetic_algorithm_enterprise_ensemble_{timestamp}.png")
        plt.savefig(ensemble_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        saved_plots['ensemble'] = ensemble_path
        print(f"    Ensemble plot saved: {ensemble_path}")

        # Save target configuration
        print("  • Generating and saving target configuration visualization...")
        plt.figure(figsize=(12, 8))
        udp.plot('target')
        target_path = os.path.join(output_dir, f"enhanced_genetic_algorithm_enterprise_target_{timestamp}.png")
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
        plt.plot(generations, best_fitness_evolution, color='darkblue', linewidth=2, label='Best Fitness Evolution',
                 marker='o')

        # Add target fitness reference line (Enterprise target)
        plt.axhline(y=-0.991, color='red', linestyle='--', linewidth=2, label='Championship Target (-0.991)')

        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.title('Enhanced Genetic Algorithm Convergence Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Create subplot for fitness distribution analysis
        plt.subplot(1, 2, 2)
        all_fitness_values = fitness_history if len(fitness_history) > len(
            best_fitness_evolution) else best_fitness_evolution

        plt.hist(all_fitness_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(all_fitness_values), color='red', linestyle='--',
                    label=f'Mean: {np.mean(all_fitness_values):.6f}')
        plt.axvline(np.min(all_fitness_values), color='green', linestyle='--',
                    label=f'Best: {np.min(all_fitness_values):.6f}')
        plt.axvline(-0.991, color='orange', linestyle=':', label='Target: -0.991')

        plt.xlabel('Fitness Value')
        plt.ylabel('Frequency')
        plt.title('Fitness Distribution Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        convergence_path = os.path.join(results_path,
                                        f"enhanced_genetic_algorithm_enterprise_convergence_{timestamp}.png")
        plt.savefig(convergence_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Convergence plot saved: {convergence_path}")
        return convergence_path

    except Exception as e:
        print(f"    Convergence plot error: {e}")
        return None
