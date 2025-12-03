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

@dataclass
class SolutionMemory:
    """Memory structure for tracking good solution patterns"""
    chromosome: List[int]
    fitness: float
    patterns: List[Tuple[int, int]]
    frequency: int = 1


class EnhancedIndividual:
    """Enhanced individual with more sophisticated features"""

    def __init__(self, chromosome=None):
        self.chromosome = chromosome if chromosome is not None else []
        self.fitness = float('inf')
        self.moves_count = 0
        self.placement_accuracy = 0.0
        self.move_efficiency = 0.0
        self.is_evaluated = False
        self.age = 0
        self.diversity_score = 0.0
        self.novelty_score = 0.0
        self.patterns = []
        self.population_id = 0

    def copy(self):
        """Create a deep copy of the individual."""
        new_individual = EnhancedIndividual(self.chromosome.copy() if self.chromosome else [])
        new_individual.fitness = self.fitness
        new_individual.moves_count = self.moves_count
        new_individual.placement_accuracy = self.placement_accuracy
        new_individual.move_efficiency = self.move_efficiency
        new_individual.is_evaluated = self.is_evaluated
        new_individual.age = self.age
        new_individual.diversity_score = self.diversity_score
        new_individual.novelty_score = self.novelty_score
        new_individual.patterns = self.patterns.copy()
        new_individual.population_id = self.population_id
        return new_individual

    def is_better_than(self, other):
        """Check if this individual is better (smaller fitness is better)."""
        return self.fitness < other.fitness

    def extract_patterns(self):
        """Extract useful move patterns from chromosome"""
        if len(self.chromosome) < 5:
            return []

        patterns = []
        end_pos = self.chromosome.index(-1) if -1 in self.chromosome else len(self.chromosome)

        # Extract 2-move patterns
        for i in range(0, end_pos - 3, 2):
            if i + 3 < end_pos:
                pattern = tuple(self.chromosome[i:i + 4])
                patterns.append(pattern)

        self.patterns = patterns
        return patterns


class TabuList:
    """Tabu list to avoid cycling through bad solutions"""

    def __init__(self, max_size=TABU_SIZE):
        self.tabu_moves = deque(maxlen=max_size)
        self.tabu_patterns = deque(maxlen=max_size)

    def add_move(self, cube_id, move_cmd):
        self.tabu_moves.append((cube_id, move_cmd))

    def add_pattern(self, pattern):
        self.tabu_patterns.append(pattern)

    def is_tabu_move(self, cube_id, move_cmd):
        return (cube_id, move_cmd) in self.tabu_moves

    def is_tabu_pattern(self, pattern):
        return pattern in self.tabu_patterns


class NoveltyArchive:
    """Archive for maintaining solution diversity"""

    def __init__(self, max_size=NOVELTY_ARCHIVE_SIZE):
        self.solutions = []
        self.max_size = max_size

    def add_solution(self, individual):
        if len(self.solutions) < self.max_size:
            self.solutions.append(individual.copy())
        else:
            # Replace least novel solution
            least_novel_idx = min(range(len(self.solutions)),
                                  key=lambda i: self.solutions[i].novelty_score)
            if individual.novelty_score > self.solutions[least_novel_idx].novelty_score:
                self.solutions[least_novel_idx] = individual.copy()

    def calculate_novelty(self, individual):
        """Calculate novelty score based on distance to archive"""
        if not self.solutions:
            return 1.0

        distances = []
        for archived in self.solutions:
            distance = self._chromosome_distance(individual.chromosome, archived.chromosome)
            distances.append(distance)

        # Average distance to k-nearest neighbors
        k = min(3, len(distances))  # Reduced for Enterprise
        distances.sort()
        novelty = np.mean(distances[:k])
        individual.novelty_score = novelty
        return novelty

    def _chromosome_distance(self, chrom1, chrom2):
        """Calculate distance between two chromosomes"""
        try:
            end1 = chrom1.index(-1) if -1 in chrom1 else len(chrom1)
            end2 = chrom2.index(-1) if -1 in chrom2 else len(chrom2)

            seq1 = chrom1[:end1]
            seq2 = chrom2[:end2]

            # Hamming distance for overlapping part
            min_len = min(len(seq1), len(seq2))
            if min_len == 0:
                return 1.0

            differences = sum(1 for i in range(min_len) if seq1[i] != seq2[i])
            length_diff = abs(len(seq1) - len(seq2))

            return (differences + length_diff) / max(len(seq1), len(seq2), 1)
        except:
            return 1.0


class SolutionMemoryBank:
    """Memory bank for storing and retrieving good solution patterns"""

    def __init__(self, max_size=MEMORY_SIZE):
        self.memory = []
        self.max_size = max_size
        self.pattern_frequency = defaultdict(int)

    def add_solution(self, individual):
        if individual.fitness == float('inf'):
            return

        patterns = individual.extract_patterns()
        memory_entry = SolutionMemory(
            chromosome=individual.chromosome.copy(),
            fitness=individual.fitness,
            patterns=patterns
        )

        # Update pattern frequencies
        for pattern in patterns:
            self.pattern_frequency[pattern] += 1

        # Add to memory
        if len(self.memory) < self.max_size:
            self.memory.append(memory_entry)
        else:
            # Replace worst solution
            worst_idx = max(range(len(self.memory)), key=lambda i: self.memory[i].fitness)
            if individual.fitness < self.memory[worst_idx].fitness:
                old_patterns = self.memory[worst_idx].patterns
                for pattern in old_patterns:
                    self.pattern_frequency[pattern] = max(0, self.pattern_frequency[pattern] - 1)

                self.memory[worst_idx] = memory_entry

    def get_good_patterns(self, top_k=8):  # Reduced for Enterprise
        """Get most frequent good patterns"""
        sorted_patterns = sorted(self.pattern_frequency.items(),
                                 key=lambda x: x[1], reverse=True)
        return [pattern for pattern, freq in sorted_patterns[:top_k] if freq > 1]

    def sample_from_memory(self):
        """Sample a good solution from memory"""
        if not self.memory:
            return None

        # Weighted sampling by fitness (better fitness = higher probability)
        fitnesses = [mem.fitness for mem in self.memory]
        if all(f == float('inf') for f in fitnesses):
            return None

        # Convert to probabilities (smaller fitness = higher probability)
        max_fitness = max(f for f in fitnesses if f != float('inf'))
        weights = [max_fitness - f + 1 for f in fitnesses]
        total_weight = sum(weights)

        if total_weight == 0:
            return random.choice(self.memory)

        probs = [w / total_weight for w in weights]
        return np.random.choice(self.memory, p=probs)


# Include all the helper functions from the ISS enhanced version, adapted for Enterprise
def advanced_remove_inverse_moves(chromosome_seq):
    """Advanced inverse move removal with pattern detection"""
    if len(chromosome_seq) < 4:
        return chromosome_seq

    cleaned_seq = []
    i = 0

    while i < len(chromosome_seq) - 1:
        if i + 3 < len(chromosome_seq):
            cube1, move1 = chromosome_seq[i], chromosome_seq[i + 1]
            cube2, move2 = chromosome_seq[i + 2], chromosome_seq[i + 3]

            # Check for direct inverse moves
            if cube1 == cube2 and are_inverse_moves(move1, move2):
                i += 4
                continue

            # Check for delayed inverse moves
            if cube1 == cube2 and move1 == move2 and i + 5 < len(chromosome_seq):
                # Look ahead for potential cycles
                found_cycle = False
                for j in range(i + 4, min(i + 10, len(chromosome_seq) - 1), 2):  # Reduced window for Enterprise
                    if j + 1 < len(chromosome_seq):
                        cube3, move3 = chromosome_seq[j], chromosome_seq[j + 1]
                        if cube3 == cube1 and move3 == move1:
                            # Found repetitive pattern, skip one occurrence
                            i += 4
                            found_cycle = True
                            break

                if found_cycle:
                    continue

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


def pattern_guided_cube_selection(udp, recent_moves, good_patterns, memory_bank, current_fitness=0.0):
    """Cube selection guided by successful patterns and memory - Enterprise optimized"""
    num_cubes = udp.setup['num_cubes']

    # For Enterprise's large cube count, use more efficient selection
    if num_cubes > 1000:
        # Sample from a subset for efficiency
        sample_size = min(200, num_cubes)
        candidate_cubes = random.sample(range(num_cubes), sample_size)
    else:
        candidate_cubes = list(range(num_cubes))

    cube_scores = np.ones(len(candidate_cubes))

    # Base scoring
    for i, cube_id in enumerate(candidate_cubes):
        score = 1.0

        # Penalty for recent moves
        recent_penalty = len(recent_moves[cube_id]) / 12
        score *= (1.0 - recent_penalty * 0.5)

        # Bonus for cubes in good patterns
        pattern_bonus = 0.0
        for pattern in good_patterns:
            if len(pattern) >= 2 and pattern[0] == cube_id:
                pattern_bonus += 0.15

        score *= (1.0 + pattern_bonus)

        # Fitness-based adaptation
        if current_fitness < -0.3:  # Good performance
            score *= (0.9 + random.random() * 0.2)  # More focused
        elif current_fitness < 0:  # Moderate performance
            score *= (0.7 + random.random() * 0.6)  # Balanced
        else:  # Poor performance
            score *= (0.4 + random.random() * 0.6)  # More exploratory

        cube_scores[i] = score

    # Weighted selection
    probabilities = cube_scores / np.sum(cube_scores)
    selected_idx = np.random.choice(len(candidate_cubes), p=probabilities)
    return candidate_cubes[selected_idx]


def pattern_guided_move_selection(cube_id, recent_moves, good_patterns, current_fitness=0.0):
    """Move selection guided by successful patterns"""
    move_scores = np.ones(6)

    for move_cmd in range(6):
        score = 1.0

        # Penalty for recent moves
        if move_cmd in recent_moves[cube_id]:
            score *= 0.3

        # Bonus for moves in good patterns
        pattern_bonus = 0.0
        for pattern in good_patterns:
            if len(pattern) >= 2 and pattern[0] == cube_id and pattern[1] == move_cmd:
                pattern_bonus += 0.25

        score *= (1.0 + pattern_bonus)

        # Fitness-based strategy
        if current_fitness < -0.3:
            # Good fitness, prefer tested moves
            if move_cmd % 2 == 1:  # Counterclockwise moves
                score *= 1.2
        elif current_fitness < 0:
            # Moderate fitness, balanced approach
            score *= (0.8 + random.random() * 0.4)
        else:
            # Poor fitness, try different approaches
            score *= (0.6 + random.random() * 0.4)

        move_scores[move_cmd] = score

    # Weighted selection
    if np.sum(move_scores) > 0:
        probabilities = move_scores / np.sum(move_scores)
        return np.random.choice(6, p=probabilities)
    else:
        return random.randint(0, 5)

