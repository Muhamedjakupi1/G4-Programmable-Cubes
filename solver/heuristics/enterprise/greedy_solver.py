import sys
import os
import numpy as np
import random
import json
import time
from datetime import datetime
from tqdm import tqdm
from scipy.spatial.distance import cdist
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for automated execution
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Matplotlib not available - plotting disabled")

# Add the src directory and the repository root to the Python path
repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))

from programmable_cubes_UDP import programmable_cubes_UDP


# Experimental Configuration Parameters
EXPERIMENTAL_ITERATIONS = 500         # Reduced for faster testing during development
MAX_CHROMOSOME_LENGTH = 10000        # Increased to match random search competitive scale
RECENT_MOVES_MEMORY = 6             # Reduced temporal memory for less restrictive move selection
RANDOM_SEED = 42                    # Deterministic seed for reproducible experiments
LOG_INTERVAL = 50                   # More frequent progress reporting for testing
EXPLORATION_FACTOR = 0.3            # Increased exploration-exploitation balance for better diversity
GREEDY_FACTOR = 0.7                 # Reduced greedy selection for more exploration
TEMPERATURE = 0.2                   # Increased temperature for enhanced stochastic behavior
RESULTS_DIR = "solver/results/enterprise"  # Academic output directory for experimental data


def save_experimental_results(results_data, filename_prefix="greedy_enterprise_experiment"):
    """
    Save comprehensive experimental results to JSON file with academic metadata.
    
    Args:
        results_data (dict): Experimental data and performance metrics
        filename_prefix (str): Base filename for results storage
        
    Returns:
        str: Filepath of saved results file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(repo_root, RESULTS_DIR)
    heuristics_path = os.path.join(results_path, "heuristics")
    os.makedirs(heuristics_path, exist_ok=True)
    
    filepath = os.path.join(heuristics_path, f"{filename_prefix}_{timestamp}.json")
    
    with open(filepath, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Experimental results saved: {filepath}")
    return filepath


def save_solution_visualizations(udp, best_chromosome, fitness_score, filename_prefix="greedy_enterprise_solution"):
    """
    Generate and save academic-quality visualizations of optimization results.
    
    Args:
        udp: UDP instance for problem access
        best_chromosome: Optimal solution chromosome
        fitness_score: Achieved fitness value
        filename_prefix: Base filename for visualization files
    """
    if not PLOTTING_AVAILABLE:
        print("Visualization capability not available - matplotlib required")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(repo_root, RESULTS_DIR)
    heuristics_path = os.path.join(results_path, "heuristics")
    os.makedirs(heuristics_path, exist_ok=True)
    
    try:
        # Evaluate solution to establish final state
        udp.fitness(best_chromosome)
        
        # Create target visualization
        plt.figure(figsize=(12, 10))
        udp.plot('target')
        plt.title(f'Enterprise Target Configuration\nGreedy Heuristic Optimization', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save target visualization
        target_filepath = os.path.join(heuristics_path, f"{filename_prefix}_target_{timestamp}.png")
        plt.savefig(target_filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Target visualization saved: {target_filepath}")
        
        # Create ensemble visualization  
        plt.figure(figsize=(12, 10))
        udp.plot('ensemble')
        plt.title(f'Enterprise Optimized Assembly Configuration\nFitness: {fitness_score:.6f}', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save ensemble visualization
        ensemble_filepath = os.path.join(heuristics_path, f"{filename_prefix}_ensemble_{timestamp}.png")
        plt.savefig(ensemble_filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Ensemble visualization saved: {ensemble_filepath}")
        
    except Exception as e:
        print(f"Visualization generation encountered error: {e}")
        import traceback
        traceback.print_exc()


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
        plt.title('Greedy Heuristic Convergence Analysis')
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
        
        convergence_path = os.path.join(output_dir, f"greedy_enterprise_convergence_{timestamp}.png")
        plt.savefig(convergence_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Convergence analysis plot saved: {convergence_path}")
        return convergence_path
        
    except Exception as e:
        print(f"Convergence plot generation error: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_cube_distances(current_positions, target_positions, cube_types, target_cube_types):
    """
    Calculate the minimum distance from each cube to its nearest target position of the same type.
    
    Args:
        current_positions (np.ndarray): Current cube positions [n_cubes, 3]
        target_positions (np.ndarray): Target cube positions [n_targets, 3]
        cube_types (np.ndarray): Types of current cubes [n_cubes]
        target_cube_types (np.ndarray): Types of target cubes [n_targets]
        
    Returns:
        np.ndarray: Minimum distances for each cube to its target type
    """
    n_cubes = len(current_positions)
    min_distances = np.zeros(n_cubes)
    
    # For each cube type, find minimum distances
    for cube_type in np.unique(cube_types):
        # Get cubes of this type
        cube_mask = cube_types == cube_type
        target_mask = target_cube_types == cube_type
        
        if np.any(cube_mask) and np.any(target_mask):
            # Calculate distances between cubes of this type and their targets
            distances = cdist(current_positions[cube_mask], target_positions[target_mask])
            # Find minimum distance for each cube
            min_distances[cube_mask] = np.min(distances, axis=1)
    
    return min_distances


def calculate_target_centroid(target_positions):
    """
    Calculate the centroid (center of mass) of the target structure.
    
    Args:
        target_positions (np.ndarray): Target cube positions [n_targets, 3]
        
    Returns:
        np.ndarray: Centroid coordinates [3]
    """
    return np.mean(target_positions, axis=0)


def select_next_cube(current_positions, target_positions, cube_types, target_cube_types, 
                    recent_moves, max_distance_threshold=0.05):
    """
    IMPROVED: Select the cube using enhanced greedy strategy with better heuristics.
    
    Args:
        current_positions (np.ndarray): Current cube positions
        target_positions (np.ndarray): Target cube positions
        cube_types (np.ndarray): Types of current cubes
        target_cube_types (np.ndarray): Types of target cubes
        recent_moves (dict): Dictionary tracking recent moves for each cube
        max_distance_threshold (float): Minimum distance to consider a cube for selection
        
    Returns:
        int: ID of the selected cube, or -1 if no suitable cube found
    """
    distances = calculate_cube_distances(current_positions, target_positions, 
                                       cube_types, target_cube_types)
    
    # Calculate target centroid for additional heuristics
    target_centroid = calculate_target_centroid(target_positions)
    
    n_cubes = len(current_positions)
    cube_scores = np.zeros(n_cubes)
    
    # Enhanced scoring system
    for cube_id in range(n_cubes):
        if distances[cube_id] > max_distance_threshold:
            # Base score: distance from target
            base_score = distances[cube_id]
            
            # Bonus: distance from centroid (prefer cubes closer to target structure)
            cube_pos = current_positions[cube_id]
            centroid_distance = np.linalg.norm(cube_pos - target_centroid)
            centroid_bonus = 1.0 / (1.0 + centroid_distance)
            
            # Penalty: recent moves (reduce redundancy)
            recent_penalty = len(recent_moves[cube_id]) * 0.1
            
            # Combine scores
            cube_scores[cube_id] = base_score + centroid_bonus - recent_penalty
    
    # Find eligible cubes
    eligible_cubes = np.where(cube_scores > 0)[0]
    
    if len(eligible_cubes) == 0:
        # Fallback: select any cube with distance > threshold
        eligible_cubes = np.where(distances > max_distance_threshold)[0]
    
    if len(eligible_cubes) == 0:
        return -1  # No suitable cube found
    
    # Enhanced selection strategy
    if np.random.random() < EXPLORATION_FACTOR:
        # Exploration: random selection with temperature-based probability
        if np.random.random() < TEMPERATURE:
            return np.random.choice(eligible_cubes)
        else:
            # Weighted random selection
            eligible_scores = cube_scores[eligible_cubes]
            if np.sum(eligible_scores) > 0:
                probabilities = eligible_scores / np.sum(eligible_scores)
                return np.random.choice(eligible_cubes, p=probabilities)
            else:
                return np.random.choice(eligible_cubes)
    else:
        # Exploitation: select best cube
        best_cube_idx = np.argmax(cube_scores[eligible_cubes])
        return eligible_cubes[best_cube_idx]