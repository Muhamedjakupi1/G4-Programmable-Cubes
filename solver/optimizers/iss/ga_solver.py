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

class Individual:
    """Individual with  fitness handling."""
    
    def __init__(self, chromosome=None):
        self.chromosome = chromosome if chromosome is not None else []
        self.fitness = float('inf')  # Start with worst possible (positive infinity)
        self.moves_count = 0
        self.placement_accuracy = 0.0
        self.move_efficiency = 0.0
        self.is_evaluated = False
        self.age = 0
    
    def copy(self):
        """Create a deep copy of the individual."""
        new_individual = Individual(self.chromosome.copy() if self.chromosome else [])
        new_individual.fitness = self.fitness
        new_individual.moves_count = self.moves_count
        new_individual.placement_accuracy = self.placement_accuracy
        new_individual.move_efficiency = self.move_efficiency
        new_individual.is_evaluated = self.is_evaluated
        new_individual.age = self.age
        return new_individual
    
    def is_better_than(self, other):
        """Check if this individual is better (: smaller fitness is better)."""
        return self.fitness < other.fitness


def remove_inverse_moves(chromosome_seq):
    """
    Remove adjacent inverse moves for the same cube to improve efficiency.
    Inverse moves: (0,1), (2,3), (4,5) are pairs that cancel each other.
    """
    if len(chromosome_seq) < 4:  # Need at least 2 moves
        return chromosome_seq
    
    cleaned_seq = []
    i = 0
    
    while i < len(chromosome_seq) - 1:
        if i + 3 < len(chromosome_seq):  # Can check next move
            # Current move
            cube1, move1 = chromosome_seq[i], chromosome_seq[i + 1]
            # Next move  
            cube2, move2 = chromosome_seq[i + 2], chromosome_seq[i + 3]
            
            # Check if same cube and inverse moves
            if cube1 == cube2 and are_inverse_moves(move1, move2):
                # Skip both moves (they cancel out)
                i += 4
                continue
        
        # Keep this move
        cleaned_seq.extend([chromosome_seq[i], chromosome_seq[i + 1]])
        i += 2
    
    return cleaned_seq


def are_inverse_moves(move1, move2):
    """Check if two moves are inverses of each other."""
    inverse_pairs = [(0, 1), (2, 3), (4, 5)]
    for pair in inverse_pairs:
        if (move1, move2) == pair or (move2, move1) == pair:
            return True
    return False


def intelligent_cube_selection(udp, recent_moves, current_fitness=0.0):
    """Intelligent cube selection with  fitness interpretation."""
    num_cubes = udp.setup['num_cubes']
    cube_scores = np.zeros(num_cubes)
    
    for cube_id in range(num_cubes):
        score = 1.0
        
        # Penalty for recent moves
        recent_penalty = len(recent_moves[cube_id]) / 8
        score *= (1.0 - recent_penalty * 0.7)
        
        # : If fitness is good (negative), be more exploitative
        if current_fitness < 0:
            # We're doing well, be more focused
            score *= (0.8 + random.random() * 0.2)
        else:
            # We're doing poorly, be more exploratory
            score *= (0.3 + random.random() * 0.7)
        
        cube_scores[cube_id] = score
    
    # Weighted selection
    probabilities = cube_scores / np.sum(cube_scores)
    return np.random.choice(num_cubes, p=probabilities)


def intelligent_move_selection(cube_id, recent_moves, current_fitness=0.0):
    """Intelligent move selection with  fitness interpretation."""
    move_scores = np.zeros(6)
    
    for move_cmd in range(6):
        score = 1.0
        
        # Penalty for recent moves
        if move_cmd in recent_moves[cube_id]:
            score *= 0.2
        
        # : Adaptive strategy based on fitness direction
        if current_fitness < 0:
            # Good fitness, prefer tested moves
            if move_cmd % 2 == 1:  # Counterclockwise generally more effective
                score *= 1.2
        else:
            # Poor fitness, try different approaches
            score *= (0.8 + random.random() * 0.4)
        
        move_scores[move_cmd] = score
    
    # Weighted selection
    if np.sum(move_scores) > 0:
        probabilities = move_scores / np.sum(move_scores)
        return np.random.choice(6, p=probabilities)
    else:
        return random.randint(0, 5)


def generate_smart_chromosome(udp, max_length):
    """Generate smart chromosome with  optimization direction."""
    chromosome = []
    length = random.randint(MIN_CHROMOSOME_LENGTH, max_length)
    recent_moves = defaultdict(list)
    
    for _ in range(length):
        cube_id = intelligent_cube_selection(udp, recent_moves)
        move_command = intelligent_move_selection(cube_id, recent_moves)
        
        chromosome.extend([cube_id, move_command])
        
        # Update recent moves
        recent_moves[cube_id].append(move_command)
        if len(recent_moves[cube_id]) > 8:
            recent_moves[cube_id].pop(0)
    
    chromosome.append(-1)
    return chromosome


def generate_greedy_chromosome(udp, max_length):
    """Generate greedy chromosome."""
    chromosome = []
    max_moves = min(max_length, max_length)
    recent_moves = defaultdict(list)
    
    for _ in range(max_moves):
        if np.random.random() < 0.8:  # 80% greedy
            cube_id = intelligent_cube_selection(udp, recent_moves)
            move_command = intelligent_move_selection(cube_id, recent_moves)
        else:
            cube_id = random.randint(0, udp.setup['num_cubes'] - 1)
            move_command = random.randint(0, 5)
        
        chromosome.extend([cube_id, move_command])
        
        recent_moves[cube_id].append(move_command)
        if len(recent_moves[cube_id]) > 8:
            recent_moves[cube_id].pop(0)
    
    chromosome.append(-1)
    return chromosome


def generate_random_chromosome(num_cubes, max_length):
    """Generate random chromosome."""
    length = random.randint(MIN_CHROMOSOME_LENGTH, max_length)
    
    chromosome = []
    for _ in range(length):
        cube_id = random.randint(0, num_cubes - 1)
        move_command = random.randint(0, 5)
        chromosome.extend([cube_id, move_command])
    
    chromosome.append(-1)
    return chromosome


def evaluate_individual(individual, udp):
    """Evaluate individual with  fitness handling."""
    if individual.is_evaluated:
        return individual.fitness
    
    try:
        chromosome_array = np.array(individual.chromosome, dtype=int)
        fitness_score = udp.fitness(chromosome_array)
        individual.fitness = fitness_score[0]  # UDP returns negative values for good solutions
        individual.moves_count = count_moves(individual.chromosome)
        
        # Calculate detailed metrics if possible
        if hasattr(udp, 'final_cube_positions') and udp.final_cube_positions is not None:
            target_positions = udp.target_cube_positions
            final_positions = udp.final_cube_positions
            target_types = udp.target_cube_types
            initial_types = udp.initial_cube_types
            
            num_correct = 0
            total_cubes = len(final_positions)
            
            for cube_type in range(udp.setup['num_cube_types']):
                target_list = target_positions[target_types == cube_type].tolist()
                final_list = final_positions[initial_types == cube_type].tolist()
                overlap = [cube in final_list for cube in target_list]
                num_correct += np.sum(overlap)
            
            individual.placement_accuracy = num_correct / total_cubes
            individual.move_efficiency = 1.0 - (individual.moves_count / udp.setup['max_cmds'])
        
        individual.is_evaluated = True
    except Exception as e:
        print(f"Error evaluating individual: {e}")
        individual.fitness = float('inf')  # Worst possible fitness
        individual.moves_count = 0
        individual.placement_accuracy = 0.0
        individual.move_efficiency = 0.0
        individual.is_evaluated = True
    
    return individual.fitness


def count_moves(chromosome):
    """Count moves in chromosome."""
    if not chromosome:
        return 0
    try:
        end_pos = chromosome.index(-1)
        return end_pos // 2
    except ValueError:
        return len(chromosome) // 2


def initialize_population(udp, population_size):
    """Initialize population with diverse strategies."""
    population = []
    num_cubes = udp.setup['num_cubes']
    
    num_smart = int(population_size * SMART_INITIALIZATION_RATIO)
    num_greedy = int(population_size * GREEDY_INITIALIZATION_RATIO)
    num_random = population_size - num_smart - num_greedy
    
    print(f" initialization: {num_smart} smart, {num_greedy} greedy, {num_random} random")
    
    # Generate smart individuals
    for _ in range(num_smart):
        chromosome = generate_smart_chromosome(udp, MAX_CHROMOSOME_LENGTH)
        individual = Individual(chromosome)
        population.append(individual)
    
    # Generate greedy individuals
    for _ in range(num_greedy):
        chromosome = generate_greedy_chromosome(udp, MAX_CHROMOSOME_LENGTH)
        individual = Individual(chromosome)
        population.append(individual)
    
    # Generate random individuals
    for _ in range(num_random):
        chromosome = generate_random_chromosome(num_cubes, MAX_CHROMOSOME_LENGTH)
        individual = Individual(chromosome)
        population.append(individual)
    
    return population


def _tournament_selection(population, tournament_size=TOURNAMENT_SIZE):
    """ tournament selection - smaller fitness is better."""
    tournament = random.sample(population, min(tournament_size, len(population)))
    # : Return individual with SMALLEST fitness
    return min(tournament, key=lambda ind: ind.fitness)


def _crossover(parent1, parent2, udp):
    """ crossover - better parent has smaller fitness."""
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()
    
    try:
        # Get sequences
        end1 = parent1.chromosome.index(-1) if -1 in parent1.chromosome else len(parent1.chromosome)
        end2 = parent2.chromosome.index(-1) if -1 in parent2.chromosome else len(parent2.chromosome)
        
        seq1 = parent1.chromosome[:end1]
        seq2 = parent2.chromosome[:end2]
        
        if len(seq1) < 4 or len(seq2) < 4:
            return parent1.copy(), parent2.copy()
        
        # : Better parent has SMALLER fitness
        better_parent = parent1 if parent1.fitness < parent2.fitness else parent2
        worse_parent = parent2 if parent1.fitness < parent2.fitness else parent1
        
        # Bias toward better parent (75% vs 25%)
        better_seq = better_parent.chromosome[:better_parent.chromosome.index(-1)]
        worse_seq = worse_parent.chromosome[:worse_parent.chromosome.index(-1)]
        
        offspring1_seq = []
        offspring2_seq = []
        
        # Segment-based crossover with bias toward better parent
        min_len = min(len(better_seq), len(worse_seq))
        for i in range(0, min_len, 2):
            if random.random() < 0.75:  # Strong bias toward better parent
                if i + 1 < len(better_seq):
                    offspring1_seq.extend(better_seq[i:i+2])
                if i + 1 < len(worse_seq):
                    offspring2_seq.extend(worse_seq[i:i+2])
            else:
                if i + 1 < len(worse_seq):
                    offspring1_seq.extend(worse_seq[i:i+2])
                if i + 1 < len(better_seq):
                    offspring2_seq.extend(better_seq[i:i+2])
        
        # Add remaining from better parent
        if len(better_seq) > min_len:
            offspring1_seq.extend(better_seq[min_len:])
        
        # Create offspring
        offspring1 = Individual(repair_chromosome(offspring1_seq, udp))
        offspring2 = Individual(repair_chromosome(offspring2_seq, udp))
        
        return offspring1, offspring2
        
    except Exception:
        return parent1.copy(), parent2.copy()

