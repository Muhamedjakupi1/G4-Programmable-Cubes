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


def repair_chromosome(chromosome_seq, udp):
    """Enhanced chromosome repair with inverse-move cleanup."""
    if not chromosome_seq:
        return generate_random_chromosome(udp.setup['num_cubes'], MAX_CHROMOSOME_LENGTH)
    
    # Ensure even length
    if len(chromosome_seq) % 2 != 0:
        chromosome_seq = chromosome_seq[:-1]
    
    # Apply inverse-move cleanup
    if random.random() < CLEANUP_RATE:
        chromosome_seq = remove_inverse_moves(chromosome_seq)
    
    # Ensure reasonable length
    if len(chromosome_seq) < MIN_CHROMOSOME_LENGTH * 2:
        num_cubes = udp.setup['num_cubes']
        recent_moves = defaultdict(list)
        
        while len(chromosome_seq) < MIN_CHROMOSOME_LENGTH * 2:
            cube_id = intelligent_cube_selection(udp, recent_moves)
            move_cmd = intelligent_move_selection(cube_id, recent_moves)
            chromosome_seq.extend([cube_id, move_cmd])
            
            recent_moves[cube_id].append(move_cmd)
            if len(recent_moves[cube_id]) > 8:
                recent_moves[cube_id].pop(0)
    
    if len(chromosome_seq) > MAX_CHROMOSOME_LENGTH * 2:
        chromosome_seq = chromosome_seq[:MAX_CHROMOSOME_LENGTH * 2]
    
    # Validate values
    num_cubes = udp.setup['num_cubes']
    for i in range(0, len(chromosome_seq), 2):
        if i + 1 < len(chromosome_seq):
            if chromosome_seq[i] < 0 or chromosome_seq[i] >= num_cubes:
                chromosome_seq[i] = random.randint(0, num_cubes - 1)
            if chromosome_seq[i + 1] < 0 or chromosome_seq[i + 1] > 5:
                chromosome_seq[i + 1] = random.randint(0, 5)
    
    result = chromosome_seq.copy()
    result.append(-1)
    return result


def adaptive_mutation(individual, udp, mutation_rate):
    """Mutation with adaptive rate based on stagnation."""
    if random.random() > mutation_rate:
        return individual
    
    mutated = individual.copy()
    chromosome = mutated.chromosome.copy()
    
    try:
        end_pos = chromosome.index(-1) if -1 in chromosome else len(chromosome)
        move_seq = chromosome[:end_pos]
    except (ValueError, AttributeError):
        move_seq = generate_random_chromosome(udp.setup['num_cubes'], MAX_CHROMOSOME_LENGTH)[:-1]
    
    if len(move_seq) < MIN_CHROMOSOME_LENGTH * 2:
        move_seq = generate_random_chromosome(udp.setup['num_cubes'], MAX_CHROMOSOME_LENGTH)[:-1]
    
    # Number of mutations based on current mutation rate
    num_mutations = max(1, int(len(move_seq) * mutation_rate / 20))
    
    for _ in range(num_mutations):
        mutation_type = random.random()
        
        if mutation_type < 0.4 and len(move_seq) < MAX_CHROMOSOME_LENGTH * 2:
            # Insert mutation
            try:
                insert_pos = random.randrange(0, len(move_seq) + 1, 2)
                new_move = [random.randint(0, udp.setup['num_cubes'] - 1), random.randint(0, 5)]
                move_seq = move_seq[:insert_pos] + new_move + move_seq[insert_pos:]
            except ValueError:
                new_move = [random.randint(0, udp.setup['num_cubes'] - 1), random.randint(0, 5)]
                move_seq.extend(new_move)
        
        elif mutation_type < 0.7 and len(move_seq) > MIN_CHROMOSOME_LENGTH * 2:
            # Delete mutation
            try:
                delete_pos = random.randrange(0, len(move_seq), 2)
                if delete_pos + 1 < len(move_seq):
                    move_seq = move_seq[:delete_pos] + move_seq[delete_pos + 2:]
            except ValueError:
                pass
        
        else:
            # Modify mutation
            if len(move_seq) >= 2:
                try:
                    modify_pos = random.randrange(0, len(move_seq), 2)
                    if modify_pos + 1 < len(move_seq):
                        if random.random() < 0.5:
                            move_seq[modify_pos] = random.randint(0, udp.setup['num_cubes'] - 1)
                        else:
                            move_seq[modify_pos + 1] = random.randint(0, 5)
                except ValueError:
                    pass
    
    # Repair and update
    mutated.chromosome = repair_chromosome(move_seq, udp)
    mutated.is_evaluated = False
    return mutated


def local_search(individual, udp, iterations=LOCAL_SEARCH_ITERATIONS):
    """Local search for fine-tuning solutions."""
    if random.random() > LOCAL_SEARCH_RATE:
        return individual
    
    best = individual.copy()
    current = individual.copy()
    
    for _ in range(iterations):
        # Small modifications
        candidate = current.copy()
        
        if len(candidate.chromosome) > 3:
            try:
                end_pos = candidate.chromosome.index(-1)
                if end_pos >= 4:
                    pos = random.randrange(0, end_pos - 1, 2)
                    if random.random() < 0.5:
                        candidate.chromosome[pos] = random.randint(0, udp.setup['num_cubes'] - 1)
                    else:
                        candidate.chromosome[pos + 1] = random.randint(0, 5)
                    
                    candidate.is_evaluated = False
                    evaluate_individual(candidate, udp)
                    
                    # : Better means smaller fitness
                    if candidate.fitness < current.fitness:
                        current = candidate.copy()
                        if current.fitness < best.fitness:
                            best = current.copy()
            except (ValueError, IndexError):
                continue
    
    return best


def genetic_algorithm_iss():
    """
    Execute Enhanced Genetic Algorithm for ISS Spacecraft Assembly Optimization.
    
    Implements a comprehensive genetic algorithm optimization with multi-strategy
    population initialization, adaptive mutation mechanisms, elite preservation,
    and systematic performance monitoring. The algorithm generates comprehensive
    experimental documentation suitable for academic research and competitive
    submission.
    
    Returns:
        tuple: (best_chromosome, best_fitness, best_moves_count)
            - best_chromosome: Optimal solution representation
            - best_fitness: Corresponding fitness value (negative indicates superior performance)
            - best_moves_count: Number of movement commands in optimal solution
    """
    print("=" * 80)
    print("Enhanced Genetic Algorithm for ISS Spacecraft Assembly Problem")
    print("Programmable Cubes Challenge - GECCO 2024 Competition")
    print("=" * 80)
    print()
    print("Algorithm Configuration:")
    print("  • Optimization Approach: Advanced Genetic Algorithm")
    print("  • Problem Domain: ISS Spacecraft Assembly")
    print("  • Fitness Direction: Negative values indicate superior solutions")
    print("  • Target Performance: Fitness ≤ -0.991 (championship level)")
    print()
    
    # Initialize experimental timing and metadata
    experiment_start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize UDP and extract problem parameters
    print("Initializing User Defined Problem (UDP) for ISS configuration...")
    udp = programmable_cubes_UDP('ISS')
    
    # Extract problem configuration parameters
    num_cubes = udp.setup['num_cubes']
    max_cmds = udp.setup['max_cmds']
    
    print(f"Problem Instance Characteristics:")
    print(f"  • Number of programmable cubes: {num_cubes}")
    print(f"  • Maximum movement commands: {max_cmds}")
    print(f"  • Population size: {POPULATION_SIZE}")
    print(f"  • Maximum generations: {GENERATIONS}")
    print(f"  • Base mutation rate: {BASE_MUTATION_RATE}")
    print(f"  • Cleanup rate: {CLEANUP_RATE}")
    print(f"  • Tournament size: {TOURNAMENT_SIZE}")
    print(f"  • Elite preservation count: {ELITE_SIZE}")
    print()
    
    # Initialize experimental data structures for comprehensive tracking
    algorithm_start_time = time.time()
    fitness_evolution_data = []
    population_diversity_data = []
    mutation_rate_history = []
    generation_times = []
    best_fitness_per_generation = []
    average_fitness_per_generation = []
    all_fitness_evaluations = []  # Track all individual fitness evaluations
    
    # Initialize population with diverse strategies
    print("Initializing diverse population using multiple strategies...")
    population = initialize_population(udp, POPULATION_SIZE)
    
    # Comprehensive initial population evaluation
    print("Conducting comprehensive initial population evaluation...")
    initial_evaluation_start = time.time()
    for individual in tqdm(population, desc="Initial Population Assessment"):
        fitness = evaluate_individual(individual, udp)
        all_fitness_evaluations.append(fitness)  # Track all evaluations
    initial_evaluation_time = time.time() - initial_evaluation_start
    
    # Sort population by fitness (negative values are superior)
    population.sort(key=lambda ind: ind.fitness)
    
    # Initialize tracking variables with corrected fitness optimization
    best_individual = population[0].copy()  # Best individual has smallest (most negative) fitness
    best_fitness_history = [best_individual.fitness]
    stagnation_count = 0
    current_mutation_rate = BASE_MUTATION_RATE
    
    # Calculate initial population statistics
    initial_fitness_values = [ind.fitness for ind in population]
    initial_average_fitness = np.mean(initial_fitness_values)
    initial_fitness_std = np.std(initial_fitness_values)
    
    print(f"Initial Population Analysis:")
    print(f"  • Best fitness: {best_individual.fitness:.6f} (moves: {best_individual.moves_count})")
    print(f"  • Average fitness: {initial_average_fitness:.6f}")
    print(f"  • Fitness standard deviation: {initial_fitness_std:.6f}")
    print(f"  • Population diversity: {len(set(initial_fitness_values))}/{POPULATION_SIZE}")
    if hasattr(best_individual, 'placement_accuracy'):
        print(f"  • Best placement accuracy: {best_individual.placement_accuracy:.3f}")
        print(f"  • Best move efficiency: {best_individual.move_efficiency:.3f}")
    print(f"  • Initial evaluation time: {initial_evaluation_time:.2f} seconds")
    print()
    
    # Store initial experimental data
    best_fitness_per_generation.append(best_individual.fitness)
    average_fitness_per_generation.append(initial_average_fitness)
    mutation_rate_history.append(current_mutation_rate)
    
    # Main evolutionary optimization loop
    print("Commencing evolutionary optimization process...")
    evolution_start_time = time.time()
    
    for generation in tqdm(range(GENERATIONS), desc="Evolutionary Generations"):
        generation_start_time = time.time()
        new_population = []
        
        # Age population for diversity management
        for ind in population:
            ind.age += 1
        
        # Elite preservation: retain best individuals (smallest fitness values)
        elite = sorted(population, key=lambda ind: ind.fitness)[:ELITE_SIZE]
        new_population.extend([ind.copy() for ind in elite])
        
        # Generate offspring through genetic operations
        while len(new_population) < POPULATION_SIZE:
            # Tournament selection for parent selection
            parent1 = _tournament_selection(population)
            parent2 = _tournament_selection(population)
            
            # Genetic crossover operation
            offspring1, offspring2 = _crossover(parent1, parent2, udp)
            
            # Adaptive mutation application
            offspring1 = adaptive_mutation(offspring1, udp, current_mutation_rate)
            offspring2 = adaptive_mutation(offspring2, udp, current_mutation_rate)
            
            # Local search enhancement
            offspring1 = local_search(offspring1, udp)
            offspring2 = local_search(offspring2, udp)
            
            new_population.extend([offspring1, offspring2])
        
        # Population size regulation
        new_population = new_population[:POPULATION_SIZE]
        
        # Comprehensive offspring evaluation
        for individual in new_population:
            if not individual.is_evaluated:
                fitness = evaluate_individual(individual, udp)
                all_fitness_evaluations.append(fitness)  # Track all evaluations
        
        # Population sorting and best individual tracking
        new_population.sort(key=lambda ind: ind.fitness)
        
        # Fitness improvement detection and stagnation management
        if new_population[0].fitness < best_individual.fitness:  # Improvement detected
            best_individual = new_population[0].copy()
            stagnation_count = 0
            if (generation + 1) % LOG_INTERVAL == 0:
                print(f"Generation {generation + 1}: New optimum found | Fitness = {best_individual.fitness:.6f}")
        else:
            stagnation_count += 1
        
        # Experimental data collection
        generation_fitness_values = [ind.fitness for ind in new_population]
        best_fitness_per_generation.append(new_population[0].fitness)
        average_fitness_per_generation.append(np.mean(generation_fitness_values))
        mutation_rate_history.append(current_mutation_rate)
        
        best_fitness_history.append(best_individual.fitness)
        population = new_population
        
        # Adaptive mutation rate adjustment for stagnation prevention
        if stagnation_count > STAGNATION_THRESHOLD:
            current_mutation_rate = min(MAX_MUTATION_RATE, current_mutation_rate * ADAPTIVE_MUTATION_FACTOR)
            if (generation + 1) % LOG_INTERVAL == 0:
                print(f"Stagnation detected | Increasing mutation rate to {current_mutation_rate:.3f}")
        else:
            current_mutation_rate = max(BASE_MUTATION_RATE, current_mutation_rate * 0.98)
        
        # Population restart mechanism for severe stagnation
        if stagnation_count > 50:
            print("Severe stagnation detected | Implementing population restart")
            new_pop = initialize_population(udp, POPULATION_SIZE - ELITE_SIZE)
            population = elite + new_pop
            stagnation_count = 0
            current_mutation_rate = BASE_MUTATION_RATE
        
        # Record generation timing
        generation_time = time.time() - generation_start_time
        generation_times.append(generation_time)
        
        # Periodic progress reporting
        if (generation + 1) % LOG_INTERVAL == 0:
            current_best = new_population[0].fitness
            current_avg = np.mean(generation_fitness_values)
            elapsed = time.time() - algorithm_start_time
            
            print(f"Generation {generation + 1:3d}: Best = {current_best:.6f} | "
                  f"Average = {current_avg:.6f} | Stagnation = {stagnation_count} | "
                  f"Mutation Rate = {current_mutation_rate:.3f} | Elapsed = {elapsed:.1f}s")
    
    # Calculate comprehensive experimental results
    total_evolution_time = time.time() - evolution_start_time
    total_experiment_time = time.time() - experiment_start_time
    
    print()
    print("=" * 80)
    print("Comprehensive Experimental Results Analysis")
    print("=" * 80)
    
    # Final performance metrics
    final_fitness = best_individual.fitness
    final_moves = best_individual.moves_count
    final_chromosome_length = len(best_individual.chromosome)
    
    print(f"Optimization Performance Metrics:")
    print(f"  • Final best fitness: {final_fitness:.6f}")
    print(f"  • Number of moves used: {final_moves}")
    print(f"  • Chromosome length: {final_chromosome_length}")
    print(f"  • Total evolution time: {total_evolution_time:.2f} seconds")
    print(f"  • Total experiment time: {total_experiment_time:.2f} seconds")
    print(f"  • Average generation time: {np.mean(generation_times):.3f} seconds")
    
    # Advanced performance analysis
    if hasattr(best_individual, 'placement_accuracy'):
        print(f"  • Final placement accuracy: {best_individual.placement_accuracy:.3f}")
        print(f"  • Final move efficiency: {best_individual.move_efficiency:.3f}")
    
    # Convergence analysis
    total_improvement = best_fitness_history[0] - final_fitness
    improvement_rate = total_improvement / GENERATIONS if GENERATIONS > 0 else 0
    
    print(f"Convergence Analysis:")
    print(f"  • Total fitness improvement: {total_improvement:.6f}")
    print(f"  • Average improvement per generation: {improvement_rate:.6f}")
    print(f"  • Final stagnation count: {stagnation_count}")
    
    # Competitive performance assessment
    target_fitness = -0.991  # Championship target
    benchmark_fitness = 0.186  # Original baseline performance
    
    print(f"Competitive Performance Assessment:")
    print(f"  • Championship target: {target_fitness:.6f}")
    print(f"  • Baseline performance: {benchmark_fitness:.6f}")
    print(f"  • Current performance: {final_fitness:.6f}")
    
    # Performance categorization and status determination
    if final_fitness <= target_fitness:
        performance_status = "CHAMPION"
        print(f"  • Status: Championship-level performance achieved")
        print(f"  • Result: Ready for competitive submission")
    elif final_fitness < -0.5:
        progress_percentage = (abs(final_fitness) / 0.991) * 100
        performance_status = "ELITE"
        print(f"  • Status: Elite performance | Progress: {progress_percentage:.1f}% toward target")
    elif final_fitness < 0:
        progress_percentage = (abs(final_fitness) / 0.991) * 100
        performance_status = "COMPETITIVE"
        print(f"  • Status: Competitive performance | Progress: {progress_percentage:.1f}% toward target")
    else:
        improvement_over_baseline = benchmark_fitness - final_fitness
        if improvement_over_baseline > 0:
            performance_status = "IMPROVED"
            print(f"  • Status: Baseline improvement | Better by {improvement_over_baseline:.6f}")
        else:
            performance_status = "EXPERIMENTAL"
            print(f"  • Status: Experimental result | Fitness: {final_fitness:.6f}")
    
    print(f"  • Final Classification: {performance_status}")
    print()
    
    # Generate comprehensive experimental results data structure
    comprehensive_results = {
        "experiment_metadata": {
            "algorithm_name": "Enhanced Genetic Algorithm",
            "problem_type": "ISS Spacecraft Assembly",
            "timestamp": timestamp,
            "total_experiment_duration_seconds": total_experiment_time,
            "evolution_duration_seconds": total_evolution_time,
            "initial_evaluation_duration_seconds": initial_evaluation_time
        },
        "algorithm_configuration": {
            "population_size": POPULATION_SIZE,
            "max_generations": GENERATIONS,
            "tournament_size": TOURNAMENT_SIZE,
            "elite_size": ELITE_SIZE,
            "crossover_rate": CROSSOVER_RATE,
            "base_mutation_rate": BASE_MUTATION_RATE,
            "max_mutation_rate": MAX_MUTATION_RATE,
            "max_chromosome_length": MAX_CHROMOSOME_LENGTH,
            "min_chromosome_length": MIN_CHROMOSOME_LENGTH,
            "local_search_rate": LOCAL_SEARCH_RATE,
            "cleanup_rate": CLEANUP_RATE,
            "stagnation_threshold": STAGNATION_THRESHOLD,
            "adaptive_mutation_factor": ADAPTIVE_MUTATION_FACTOR
        },
        "problem_configuration": {
            "number_of_cubes": num_cubes,
            "maximum_commands": max_cmds,
            "target_fitness": target_fitness,
            "baseline_fitness": benchmark_fitness
        },
        "optimization_results": {
            "final_best_fitness": final_fitness,
            "final_moves_count": final_moves,
            "final_chromosome_length": final_chromosome_length,
            "total_fitness_improvement": total_improvement,
            "average_improvement_per_generation": improvement_rate,
            "final_stagnation_count": stagnation_count,
            "performance_status": performance_status,
            "achieved_target": final_fitness <= target_fitness,
            "improvement_over_baseline": benchmark_fitness - final_fitness
        },
        "convergence_data": {
            "best_fitness_evolution": best_fitness_per_generation,
            "average_fitness_evolution": average_fitness_per_generation,
            "mutation_rate_history": mutation_rate_history,
            "generation_times": generation_times
        },
        "solution_details": {
            "best_chromosome": best_individual.chromosome,
            "chromosome_preview": best_individual.chromosome[:30] if len(best_individual.chromosome) > 30 else best_individual.chromosome
        }
    }
    
    # Add detailed metrics if available
    if hasattr(best_individual, 'placement_accuracy'):
        comprehensive_results["optimization_results"]["placement_accuracy"] = best_individual.placement_accuracy
        comprehensive_results["optimization_results"]["move_efficiency"] = best_individual.move_efficiency
    
    # Save comprehensive experimental results
    print(f"Saving comprehensive experimental results...")
    results_file_path = save_experimental_results(comprehensive_results)
    
    # Generate and save academic visualizations
    print(f"Generating visualization plots...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(repo_root, RESULTS_DIR)
    os.makedirs(results_path, exist_ok=True)
    saved_plots = save_solution_visualizations(udp, best_individual.chromosome, results_path, timestamp)
    
    # Generate and save convergence analysis plot
    convergence_plot_path = save_convergence_plot(
        all_fitness_evaluations, 
        best_fitness_per_generation, 
        results_path, 
        timestamp
    )
    
    print()
    print("Experimental Documentation Summary:")
    if results_file_path:
        print(f"  • Results file: {os.path.basename(results_file_path)}")
    if convergence_plot_path:
        print(f"  • Convergence plot: {os.path.basename(convergence_plot_path)}")
    print(f"  • Solution visualizations: Generated in results directory")
    print()
    
    print("Optimal solution chromosome preview (first 30 elements):")
    chromosome_preview = best_individual.chromosome[:30] if len(best_individual.chromosome) > 30 else best_individual.chromosome
    print(f"  {chromosome_preview}" + ("..." if len(best_individual.chromosome) > 30 else ""))
    
    print()
    print("=" * 80)
    print("Enhanced Genetic Algorithm Optimization Completed")
    print(f"Performance Classification: {performance_status}")
    print("Comprehensive experimental documentation generated")
    print("=" * 80)
    
    return best_individual.chromosome, best_individual.fitness, best_individual.moves_count
    
    # Initialize population
    print("Initializing  population...")
    population = initialize_population(udp, POPULATION_SIZE)
    
    # Evaluate initial population
    print("Evaluating initial population...")
    for individual in tqdm(population, desc="Initial Evaluation"):
        evaluate_individual(individual, udp)
    
    # : Sort by fitness (smallest first)
    population.sort(key=lambda ind: ind.fitness)
    
    best_individual = population[0].copy()  # First element is now the best
    best_fitness_history = [best_individual.fitness]
    stagnation_count = 0
    current_mutation_rate = BASE_MUTATION_RATE
    
    print(f"Initial best fitness: {best_individual.fitness:.6f} (moves: {best_individual.moves_count})")
    if hasattr(best_individual, 'placement_accuracy'):
        print(f"Initial placement accuracy: {best_individual.placement_accuracy:.3f}")
    print()
    
    # Main evolution loop
    print("Starting  evolution...")
    for generation in tqdm(range(GENERATIONS), desc=" Evolution"):
        new_population = []
        
        # Age population
        for ind in population:
            ind.age += 1
        
        #  elitism: Keep the best (smallest fitness)
        elite = sorted(population, key=lambda ind: ind.fitness)[:ELITE_SIZE]
        new_population.extend([ind.copy() for ind in elite])
        
        # Generate offspring
        while len(new_population) < POPULATION_SIZE:
            #  selection
            parent1 = _tournament_selection(population)
            parent2 = _tournament_selection(population)
            
            #  crossover
            offspring1, offspring2 = _crossover(parent1, parent2, udp)
            
            # Adaptive mutation
            offspring1 = adaptive_mutation(offspring1, udp, current_mutation_rate)
            offspring2 = adaptive_mutation(offspring2, udp, current_mutation_rate)
            
            # Local search
            offspring1 = local_search(offspring1, udp)
            offspring2 = local_search(offspring2, udp)
            
            # Reset age
            offspring1.age = 0
            offspring2.age = 0
            
            new_population.extend([offspring1, offspring2])
        
        # Trim population
        new_population = new_population[:POPULATION_SIZE]
        
        # Evaluate new individuals
        for individual in new_population:
            if not individual.is_evaluated:
                evaluate_individual(individual, udp)
        
        # : Sort by fitness (smallest first)
        new_population.sort(key=lambda ind: ind.fitness)
        
        # : Check for improvement (smaller is better)
        if new_population[0].fitness < best_individual.fitness:
            improvement = best_individual.fitness - new_population[0].fitness
            best_individual = new_population[0].copy()
            stagnation_count = 0
            current_mutation_rate = BASE_MUTATION_RATE  # Reset mutation rate
            
            # Early stopping if we achieve target
            if best_individual.fitness <= -0.99:
                print(f"\n🎯 TARGET ACHIEVED! Fitness: {best_individual.fitness:.6f}")
                break
        else:
            stagnation_count += 1
            
            # Adaptive mutation: increase rate on stagnation
            if stagnation_count > STAGNATION_THRESHOLD:
                current_mutation_rate = min(MAX_MUTATION_RATE, 
                                           BASE_MUTATION_RATE * (ADAPTIVE_MUTATION_FACTOR ** (stagnation_count - STAGNATION_THRESHOLD)))
        
        best_fitness_history.append(best_individual.fitness)
        population = new_population
        
        # Restart if severely stagnated
        if stagnation_count > 50:
            print(f"\nSevere stagnation at generation {generation + 1}. Restarting...")
            # Keep top 3, regenerate rest
            super_elite = sorted(population, key=lambda ind: ind.fitness)[:3]
            new_population = initialize_population(udp, POPULATION_SIZE - 3)
            
            for ind in new_population:
                evaluate_individual(ind, udp)
            
            population = super_elite + new_population
            population.sort(key=lambda ind: ind.fitness)
            stagnation_count = 0
            current_mutation_rate = BASE_MUTATION_RATE
        
        # Logging
        if (generation + 1) % LOG_INTERVAL == 0:
            avg_fitness = np.mean([ind.fitness for ind in population])
            diversity = np.std([ind.fitness for ind in population])
            elapsed = time.time() - start_time
            
            print(f"Gen {generation + 1:3d}: Best = {best_individual.fitness:.6f}, "
                  f"Avg = {avg_fitness:.6f}, Div = {diversity:.6f}")
            print(f"         Moves = {best_individual.moves_count}, "
                  f"Stagnation = {stagnation_count}, MutRate = {current_mutation_rate:.3f}")
            
            if hasattr(best_individual, 'placement_accuracy'):
                print(f"         Accuracy = {best_individual.placement_accuracy:.3f}, "
                      f"Time = {elapsed:.1f}s")
            
            # Progress analysis
            if best_individual.fitness < 0:
                progress = (abs(best_individual.fitness) / 0.991) * 100
                print(f"         Progress toward -0.991: {progress:.1f}%")
            print()
    
    print()
    print("=" * 60)
    print("🔧  RESULTS")
    print("=" * 60)
    
    elapsed_time = time.time() - start_time
    
    # Final results
    print(f"Best fitness achieved: {best_individual.fitness:.6f}")
    print(f"Number of moves used: {best_individual.moves_count}")
    print(f"Chromosome length: {len(best_individual.chromosome)}")
    print(f"Total evolution time: {elapsed_time:.1f} seconds")
    
    # Detailed analysis
    if hasattr(best_individual, 'placement_accuracy'):
        print(f"Placement accuracy: {best_individual.placement_accuracy:.3f}")
        print(f"Move efficiency: {best_individual.move_efficiency:.3f}")
    
    # Target analysis
    target_fitness = -0.991
    current_fitness = best_individual.fitness
    
    if current_fitness <= target_fitness:
        print(f"🎯 TARGET ACHIEVED! Exceeded first place performance!")
        print(f"✅ Ready for championship submission!")
        status = "CHAMPION"
    elif current_fitness < -0.5:
        progress = (abs(current_fitness) / 0.991) * 100
        print(f"🏆 EXCELLENT PERFORMANCE! Progress: {progress:.1f}%")
        status = "ELITE"
    elif current_fitness < 0:
        progress = (abs(current_fitness) / 0.991) * 100
        print(f"✅ NEGATIVE FITNESS ACHIEVED! Progress: {progress:.1f}%")
        status = "COMPETITIVE"
    else:
        improvement = 0.186 - current_fitness  # Compare to original 9th place
        if improvement > 0:
            print(f"📈 IMPROVEMENT! Better by {improvement:.6f}")
            status = "IMPROVED"
        else:
            print(f"⚠️ Still needs work. Current: {current_fitness:.6f}")
            status = "NEEDS WORK"
    
    print(f"🏆 Status: {status}")
    
    # Show improvement
    improvement_total = best_fitness_history[0] - best_individual.fitness
    print(f"Total improvement: {improvement_total:.6f}")
    
    print()
    print("Best chromosome (first 30 elements):")
    print(best_individual.chromosome[:30], "..." if len(best_individual.chromosome) > 30 else "")
    
    print()
    print("🔧  genetic algorithm completed!")
    
    return best_individual.chromosome, best_individual.fitness, best_individual.moves_count


if __name__ == "__main__":
    best_chromosome, best_fitness, best_moves = genetic_algorithm_iss()
