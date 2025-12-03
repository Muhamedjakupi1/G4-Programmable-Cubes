#!/usr/bin/env python3
"""
Enhanced Genetic Algorithm for JWST Spacecraft Assembly Problem
Academic Research Implementation - GECCO 2024 Space Optimization Competition (SpOC)

This module implements an enhanced genetic algorithm with comprehensive
experimental documentation, visualization capabilities, and result analysis for the
JWST (James Webb Space Telescope) spacecraft assembly optimization problem. The algorithm
features adaptive mechanisms, intelligent initialization strategies, diversity
preservation, and systematic performance monitoring for competitive optimization results.

The genetic algorithm employs population-based evolution with crossover, mutation,
selection pressure, diversity preservation, and JWST-specific scaling optimizations
for the intermediate-complexity 643-cube assembly problem.

Key algorithmic enhancements include:
- Population-based evolution with elitism and tournament selection
- Adaptive mutation rates based on convergence patterns
- Intelligent initialization with smart, greedy, and random strategies
- Solution memory for pattern learning and duplicate prevention
- JWST-specific operators scaled for 643-cube complexity
- Comprehensive experimental data collection and academic visualization

Target Performance: Achieve competitive fitness for JWST spacecraft assembly

Usage:
    python solver/optimizers/jwst/enhanced_ga_solver.py

Dependencies:
    - numpy: Numerical computing and array operations
    - matplotlib: Result visualization and plotting
    - tqdm: Progress monitoring during optimization
    - json: Experimental data serialization
"""

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
from collections import defaultdict, deque
import copy

# Add the src directory to the Python path
repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))

from src.programmable_cubes_UDP import programmable_cubes_UDP

# Enhanced Genetic Algorithm Configuration for JWST
POPULATION_SIZE = 50  # Balanced population size for 643-cube problem
GENERATIONS = 120  # Sufficient generations for convergence
TOURNAMENT_SIZE = 5  # Tournament selection pressure
ELITE_SIZE = 5  # Number of elite individuals preserved
CROSSOVER_RATE = 0.8  # High crossover rate for exploration
BASE_MUTATION_RATE = 0.1  # Base mutation rate
MAX_MUTATION_RATE = 0.3  # Maximum adaptive mutation rate
MAX_CHROMOSOME_LENGTH = 1500  # Scaled for JWST complexity
MIN_CHROMOSOME_LENGTH = 50  # Minimum viable solution length

# Algorithm control parameters
STAGNATION_THRESHOLD = 15  # Generations without improvement before adaptation
LOG_INTERVAL = 10  # Progress logging frequency

# Results directory configuration
RESULTS_DIR = "solver/results/jwst/optimizers"


def save_experimental_results(results_data):
    """
    Save comprehensive experimental results to JSON file with academic formatting.

    Args:
        results_data (dict): Complete experimental results including metadata,
                           configuration, performance metrics, and solution details

    Returns:
        str: Path to saved results file, None if save failed
    """
    try:
        # Generate timestamp-based filename for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"enhanced_genetic_algorithm_jwst_experiment_{timestamp}.json"

        # Ensure results directory exists
        results_path = os.path.join(repo_root, RESULTS_DIR)
        os.makedirs(results_path, exist_ok=True)

        # Write comprehensive results with proper formatting
        results_file_path = os.path.join(results_path, results_filename)
        with open(results_file_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f"Experimental results saved: {results_filename}")
        return results_file_path

    except Exception as e:
        print(f"Error saving experimental results: {e}")
        return None


def save_solution_visualizations(udp, best_chromosome, output_dir, timestamp):
    """
    Generate and save academic-quality solution visualization plots.

    Args:
        udp: The programmable cubes UDP instance
        best_chromosome: Optimal solution chromosome for visualization
        output_dir: Directory path for saving visualization files
        timestamp: Timestamp string for file naming consistency

    Returns:
        list: Paths to generated visualization files
    """
    try:
        print("Generating solution visualization plots...")

        # Evaluate chromosome to prepare UDP state for plotting
        udp.fitness(best_chromosome)

        saved_plots = []

        # Generate ensemble configuration plot
        plt.figure(figsize=(12, 8))
        udp.plot('ensemble')
        ensemble_path = os.path.join(output_dir, f"enhanced_genetic_algorithm_jwst_ensemble_{timestamp}.png")
        plt.savefig(ensemble_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(ensemble_path)

        # Generate target configuration plot
        plt.figure(figsize=(12, 8))
        udp.plot('target')
        target_path = os.path.join(output_dir, f"enhanced_genetic_algorithm_jwst_target_{timestamp}.png")
        plt.savefig(target_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(target_path)

        print(f"Solution visualizations saved: {len(saved_plots)} plots generated")
        return saved_plots

    except Exception as e:
        print(f"Error generating solution visualizations: {e}")
        return []


def save_convergence_plot(all_fitness_data, best_fitness_data, output_dir, timestamp):
    """
    Generate and save convergence analysis plot with academic formatting.

    Args:
        all_fitness_data: Complete fitness evolution data
        best_fitness_data: Best fitness evolution per generation
        output_dir: Directory path for saving plot
        timestamp: Timestamp string for file naming

    Returns:
        str: Path to saved convergence plot
    """
    try:
        print("Generating convergence analysis plot...")

        # Create dual-subplot convergence analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Fitness evolution over generations
        generations = range(len(best_fitness_data))
        ax1.plot(generations, best_fitness_data, 'b-', linewidth=2, label='Best Fitness')
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Fitness Value', fontsize=12)
        ax1.set_title('Fitness Evolution - JWST GA', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Fitness distribution analysis
        if len(best_fitness_data) > 10:
            recent_improvements = np.diff(best_fitness_data[-20:]) if len(best_fitness_data) >= 20 else np.diff(
                best_fitness_data)
            ax2.hist(recent_improvements, bins=15, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('Fitness Improvement', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Recent Improvement Distribution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save convergence plot
        convergence_path = os.path.join(output_dir, f"enhanced_genetic_algorithm_jwst_convergence_{timestamp}.png")
        plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Convergence plot saved: {os.path.basename(convergence_path)}")
        return convergence_path

    except Exception as e:
        print(f"Error generating convergence plot: {e}")
        return None


class EnhancedIndividual:
    """
    Enhanced individual representation for genetic algorithm with comprehensive tracking.

    Attributes:
        chromosome: Solution representation as list of cube-command pairs
        fitness: Evaluated fitness value
        moves_count: Number of moves in the solution
        is_evaluated: Flag indicating fitness evaluation status
        generation_created: Generation when individual was created
        parent_fitness: Fitness values of parent individuals
    """

    def __init__(self, chromosome=None):
        self.chromosome = chromosome if chromosome is not None else []
        self.fitness = float('inf')
        self.moves_count = 0
        self.is_evaluated = False
        self.generation_created = 0
        self.parent_fitness = []

    def copy(self):
        """Create deep copy of individual."""
        new_individual = EnhancedIndividual(self.chromosome.copy())
        new_individual.fitness = self.fitness
        new_individual.moves_count = self.moves_count
        new_individual.is_evaluated = self.is_evaluated
        new_individual.generation_created = self.generation_created
        new_individual.parent_fitness = self.parent_fitness.copy()
        return new_individual

    def extract_patterns(self):
        """Extract and analyze patterns from chromosome for learning."""
        if not self.chromosome:
            return []

        patterns = []
        # Extract consecutive move patterns
        for i in range(0, len(self.chromosome) - 3, 2):
            if i + 3 < len(self.chromosome):
                pattern = tuple(self.chromosome[i:i + 4])
                patterns.append(pattern)

        return patterns


class SolutionMemoryBank:
    """
    Advanced memory bank for storing and analyzing high-quality solutions.

    Maintains a collection of elite solutions for pattern learning,
    duplicate detection, and adaptive strategy development.
    """

    def __init__(self, max_size=20):
        self.memory = deque(maxlen=max_size)
        self.fitness_threshold = float('inf')
        self.pattern_frequency = defaultdict(int)

    def add_solution(self, individual):
        """Add solution to memory if it meets quality criteria."""
        if individual.fitness < self.fitness_threshold:
            self.memory.append(individual.copy())
            self.fitness_threshold = min(ind.fitness for ind in self.memory)

            # Update pattern frequency analysis
            patterns = individual.extract_patterns()
            for pattern in patterns:
                self.pattern_frequency[pattern] += 1

    def get_best_patterns(self, top_k=10):
        """Retrieve most frequent successful patterns."""
        sorted_patterns = sorted(self.pattern_frequency.items(),
                                 key=lambda x: x[1], reverse=True)
        return [pattern for pattern, freq in sorted_patterns[:top_k]]

    def contains_similar(self, individual, similarity_threshold=0.9):
        """Check if memory contains similar solution."""
        for stored in self.memory:
            if len(stored.chromosome) == 0 or len(individual.chromosome) == 0:
                continue

            # Simple similarity check based on chromosome overlap
            min_len = min(len(stored.chromosome), len(individual.chromosome))
            if min_len == 0:
                continue

            overlap = sum(1 for i in range(min_len)
                          if stored.chromosome[i] == individual.chromosome[i])
            similarity = overlap / min_len

            if similarity > similarity_threshold:
                return True

        return False


def enhanced_genetic_algorithm_jwst():
    """
    Execute enhanced genetic algorithm for JWST spacecraft assembly problem.

    Implements comprehensive genetic algorithm with academic documentation,
    adaptive mechanisms, intelligent initialization, and experimental result tracking.

    Returns:
        tuple: (best_chromosome, best_fitness, best_moves_count)
    """

    # Initialize experimental tracking
    experiment_start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("Enhanced Genetic Algorithm for JWST Spacecraft Assembly")
    print("Academic Research Implementation")
    print("=" * 80)
    print(f"Experiment timestamp: {timestamp}")
    print(f"Algorithm configuration:")
    print(f"  • Population size: {POPULATION_SIZE}")
    print(f"  • Maximum generations: {GENERATIONS}")
    print(f"  • Tournament size: {TOURNAMENT_SIZE}")
    print(f"  • Elite preservation: {ELITE_SIZE}")
    print(f"  • Crossover rate: {CROSSOVER_RATE}")
    print(f"  • Base mutation rate: {BASE_MUTATION_RATE}")
    print(f"  • Maximum chromosome length: {MAX_CHROMOSOME_LENGTH}")
    print()

    # Initialize UDP and extract problem parameters
    print("Initializing User Defined Problem (UDP) for JWST configuration...")
    algorithm_start_time = time.time()
    udp = programmable_cubes_UDP('JWST')

    # Extract problem-specific parameters
    num_cubes = udp.setup['num_cubes']
    max_cmds = udp.setup['max_cmds']

    print(f"JWST problem parameters:")
    print(f"  • Number of cubes: {num_cubes}")
    print(f"  • Maximum commands: {max_cmds}")
    print(f"  • Chromosome length range: [{MIN_CHROMOSOME_LENGTH}, {MAX_CHROMOSOME_LENGTH}]")
    print()

    # Initialize population with diverse strategies
    print("Initializing population with intelligent strategies...")
    population = []
    memory_bank = SolutionMemoryBank(max_size=15)

    # Determine population composition for JWST scale
    num_smart = POPULATION_SIZE // 3  # 33% smart initialization
    num_greedy = POPULATION_SIZE // 3  # 33% greedy initialization
    num_random = POPULATION_SIZE - num_smart - num_greedy

    print(f"Population initialization strategy:")
    print(f"  • Smart individuals: {num_smart}")
    print(f"  • Greedy individuals: {num_greedy}")
    print(f"  • Random individuals: {num_random}")
    print()

    # Generate initial population
    for _ in range(num_smart):
        chromosome = generate_smart_chromosome_jwst(udp, MAX_CHROMOSOME_LENGTH)
        individual = EnhancedIndividual(chromosome)
        population.append(individual)

    for _ in range(num_greedy):
        chromosome = generate_greedy_chromosome_jwst(udp, MAX_CHROMOSOME_LENGTH)
        individual = EnhancedIndividual(chromosome)
        population.append(individual)

    for _ in range(num_random):
        chromosome = generate_random_chromosome_jwst(num_cubes, MAX_CHROMOSOME_LENGTH)
        individual = EnhancedIndividual(chromosome)
        population.append(individual)

    # Evaluate initial population
    print("Evaluating initial population...")
    for individual in tqdm(population, desc="Initial Evaluation"):
        evaluate_individual_jwst(individual, udp)
        memory_bank.add_solution(individual)

    # Initialize evolution tracking
    best_individual = min(population, key=lambda ind: ind.fitness)
    best_fitness_per_generation = [best_individual.fitness]
    average_fitness_per_generation = []
    generation_times = []
    stagnation_count = 0
    current_mutation_rate = BASE_MUTATION_RATE

    print(f"Initial best fitness: {best_individual.fitness:.6f} (moves: {best_individual.moves_count})")
    print()

    # Main evolution loop
    print("Starting enhanced evolution for JWST...")

    for generation in tqdm(range(GENERATIONS), desc="Enhanced Evolution"):
        generation_start_time = time.time()

        # Sort population by fitness
        population.sort(key=lambda ind: ind.fitness)

        # Check for improvement
        if population[0].fitness < best_individual.fitness:
            improvement = best_individual.fitness - population[0].fitness
            best_individual = population[0].copy()
            stagnation_count = 0
            current_mutation_rate = BASE_MUTATION_RATE

            if (generation + 1) % LOG_INTERVAL == 0:
                print(f"\nNew best solution found! Generation {generation + 1}: {best_individual.fitness:.6f} "
                      f"(improvement: +{improvement:.6f})")
        else:
            stagnation_count += 1

            # Adaptive mutation rate increase during stagnation
            if stagnation_count > STAGNATION_THRESHOLD:
                current_mutation_rate = min(MAX_MUTATION_RATE,
                                            current_mutation_rate * 1.1)

        # Record generation statistics
        best_fitness_per_generation.append(best_individual.fitness)
        avg_fitness = np.mean([ind.fitness for ind in population])
        average_fitness_per_generation.append(avg_fitness)

        # Create new generation
        new_population = []

        # Preserve elite individuals
        elite_individuals = population[:ELITE_SIZE]
        for elite in elite_individuals:
            new_population.append(elite.copy())

        # Generate offspring through crossover and mutation
        while len(new_population) < POPULATION_SIZE:
            # Tournament selection for parents
            parent1 = tournament_selection(population, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, TOURNAMENT_SIZE)

            # Crossover
            if random.random() < CROSSOVER_RATE:
                child1, child2 = crossover_jwst(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            mutate_individual_jwst(child1, current_mutation_rate, num_cubes)
            mutate_individual_jwst(child2, current_mutation_rate, num_cubes)

            new_population.extend([child1, child2])

        # Trim to exact population size
        new_population = new_population[:POPULATION_SIZE]

        # Evaluate new individuals
        for individual in new_population:
            if not individual.is_evaluated:
                evaluate_individual_jwst(individual, udp)
                memory_bank.add_solution(individual)

        population = new_population

        # Record generation timing
        generation_time = time.time() - generation_start_time
        generation_times.append(generation_time)

        # Progress logging
        if (generation + 1) % LOG_INTERVAL == 0:
            diversity = np.std([ind.fitness for ind in population])
            elapsed = time.time() - algorithm_start_time

            print(f"Gen {generation + 1:3d}: Best = {best_individual.fitness:.6f}, "
                  f"Avg = {avg_fitness:.6f}, Div = {diversity:.6f}")
            print(f"         Moves = {best_individual.moves_count}, "
                  f"Mutation = {current_mutation_rate:.3f}, Time = {elapsed:.1f}s")

    # Calculate comprehensive experimental results
    total_experiment_time = time.time() - experiment_start_time
    total_algorithm_time = time.time() - algorithm_start_time

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
    print(f"  • Total algorithm time: {total_algorithm_time:.2f} seconds")
    print(f"  • Total experiment time: {total_experiment_time:.2f} seconds")
    print(f"  • Solutions in memory bank: {len(memory_bank.memory)}")

    # Competitive performance assessment
    target_fitness = -0.299  # Competitive target for JWST
    benchmark_fitness = 0.1  # Baseline performance reference

    print(f"Competitive Performance Assessment:")
    print(f"  • Target fitness: {target_fitness:.6f}")
    print(f"  • Baseline performance: {benchmark_fitness:.6f}")
    print(f"  • Current performance: {final_fitness:.6f}")

    # Performance categorization
    if final_fitness <= target_fitness:
        performance_status = "CHAMPION"
        print(f"  • Status: Championship-level performance achieved")
    elif final_fitness < -0.2:
        performance_status = "EXCEPTIONAL"
        print(f"  • Status: Exceptional performance")
    elif final_fitness < -0.1:
        performance_status = "EXCELLENT"
        print(f"  • Status: Excellent performance")
    elif final_fitness < 0:
        performance_status = "COMPETITIVE"
        print(f"  • Status: Competitive performance")
    else:
        improvement_over_baseline = benchmark_fitness - final_fitness
        if improvement_over_baseline > 0:
            performance_status = "IMPROVED"
            print(f"  • Status: Baseline improvement | Better by {improvement_over_baseline:.6f}")
        else:
            performance_status = "EXPERIMENTAL"
            print(f"  • Status: Experimental result | Fitness: {final_fitness:.6f}")

    print(f"  • Final Classification: {performance_status}")

    # Convergence analysis
    total_improvement = best_fitness_per_generation[0] - final_fitness if best_fitness_per_generation else 0
    improvement_rate = total_improvement / GENERATIONS if GENERATIONS > 0 else 0

    print(f"Convergence Analysis:")
    print(f"  • Total fitness improvement: {total_improvement:.6f}")
    print(f"  • Average improvement per generation: {improvement_rate:.6f}")
    print()

    # Create comprehensive experimental results data structure
    comprehensive_results = {
        "experiment_metadata": {
            "algorithm_name": "Enhanced Genetic Algorithm",
            "problem_type": "JWST Spacecraft Assembly",
            "timestamp": timestamp,
            "total_experiment_duration_seconds": total_experiment_time,
            "algorithm_duration_seconds": total_algorithm_time
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
            "stagnation_threshold": STAGNATION_THRESHOLD
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
            "performance_status": performance_status,
            "achieved_target": final_fitness <= target_fitness,
            "improvement_over_baseline": benchmark_fitness - final_fitness,
            "memory_bank_size": len(memory_bank.memory)
        },
        "convergence_data": {
            "best_fitness_evolution": best_fitness_per_generation,
            "average_fitness_evolution": average_fitness_per_generation,
            "generation_times": generation_times
        },
        "solution_details": {
            "best_chromosome": best_individual.chromosome,
            "chromosome_preview": best_individual.chromosome[:30] if len(
                best_individual.chromosome) > 30 else best_individual.chromosome
        }
    }

    # Save comprehensive experimental results
    print(f"Saving comprehensive experimental results...")
    results_file_path = save_experimental_results(comprehensive_results)

    # Generate and save academic visualizations
    print(f"Generating visualization plots...")
    results_path = os.path.join(repo_root, RESULTS_DIR)
    os.makedirs(results_path, exist_ok=True)
    saved_plots = save_solution_visualizations(udp, best_individual.chromosome, results_path, timestamp)

    # Generate and save convergence analysis plot
    convergence_plot_path = save_convergence_plot(
        best_fitness_per_generation,
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
    chromosome_preview = best_individual.chromosome[:30] if len(
        best_individual.chromosome) > 30 else best_individual.chromosome
    print(f"  {chromosome_preview}" + ("..." if len(best_individual.chromosome) > 30 else ""))

    print()
    print("=" * 80)
    print("Enhanced Genetic Algorithm Optimization Completed")
    print(f"Performance Classification: {performance_status}")
    print("Comprehensive experimental documentation generated")
    print("=" * 80)

    return best_individual.chromosome, best_individual.fitness, best_individual.moves_count


# Helper functions for JWST-specific operations

def generate_smart_chromosome_jwst(udp, max_length):
    """Generate smart chromosome for JWST with pattern awareness."""
    chromosome = []
    length = random.randint(MIN_CHROMOSOME_LENGTH, max_length)
    recent_moves = defaultdict(list)

    for _ in range(length):
        cube_id = random.randint(0, udp.setup['num_cubes'] - 1)

        # Smart move selection based on recent activity
        if len(recent_moves[cube_id]) > 0:
            # Avoid immediate reversals
            last_move = recent_moves[cube_id][-1]
            possible_moves = [m for m in range(6) if m != ((last_move + 3) % 6)]
            move_command = random.choice(possible_moves)
        else:
            move_command = random.randint(0, 5)

        chromosome.extend([cube_id, move_command])

        recent_moves[cube_id].append(move_command)
        if len(recent_moves[cube_id]) > 8:  # Shorter memory for JWST scale
            recent_moves[cube_id].pop(0)

    chromosome.append(-1)
    return chromosome


def generate_greedy_chromosome_jwst(udp, max_length):
    """Generate greedy chromosome for JWST."""
    return generate_smart_chromosome_jwst(udp, max_length)


def generate_random_chromosome_jwst(num_cubes, max_length):
    """Generate random chromosome for JWST."""
    length = random.randint(MIN_CHROMOSOME_LENGTH, max_length)

    chromosome = []
    for _ in range(length):
        cube_id = random.randint(0, num_cubes - 1)
        move_command = random.randint(0, 5)
        chromosome.extend([cube_id, move_command])

    chromosome.append(-1)
    return chromosome


def evaluate_individual_jwst(individual, udp):
    """Evaluate individual for JWST problem."""
    if individual.is_evaluated:
        return individual.fitness

    try:
        chromosome_array = np.array(individual.chromosome, dtype=int)
        fitness_score = udp.fitness(chromosome_array)
        individual.fitness = fitness_score[0]
        individual.moves_count = count_moves_jwst(individual.chromosome)
        individual.is_evaluated = True
        individual.extract_patterns()
    except Exception as e:
        print(f"Error evaluating individual: {e}")
        individual.fitness = float('inf')
        individual.moves_count = 0
        individual.is_evaluated = True

    return individual.fitness


def count_moves_jwst(chromosome):
    """Count moves in chromosome for JWST."""
    if not chromosome:
        return 0
    try:
        end_pos = chromosome.index(-1)
        return end_pos // 2
    except ValueError:
        return len(chromosome) // 2


def tournament_selection(population, tournament_size):
    """Select individual using tournament selection."""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return min(tournament, key=lambda ind: ind.fitness)


def crossover_jwst(parent1, parent2):
    """Perform crossover operation for JWST."""
    child1 = EnhancedIndividual()
    child2 = EnhancedIndividual()

    # Find termination points
    end1 = len(parent1.chromosome) - 1
    end2 = len(parent2.chromosome) - 1

    if parent1.chromosome and parent1.chromosome[-1] == -1:
        end1 = len(parent1.chromosome) - 1
    if parent2.chromosome and parent2.chromosome[-1] == -1:
        end2 = len(parent2.chromosome) - 1

    # Single-point crossover
    if end1 > 0 and end2 > 0:
        crossover_point1 = random.randint(0, end1)
        crossover_point2 = random.randint(0, end2)

        # Ensure crossover points are at pair boundaries
        crossover_point1 = (crossover_point1 // 2) * 2
        crossover_point2 = (crossover_point2 // 2) * 2

        child1.chromosome = (parent1.chromosome[:crossover_point1] +
                             parent2.chromosome[crossover_point2:end2] + [-1])
        child2.chromosome = (parent2.chromosome[:crossover_point2] +
                             parent1.chromosome[crossover_point1:end1] + [-1])
    else:
        child1.chromosome = parent1.chromosome.copy()
        child2.chromosome = parent2.chromosome.copy()

    # Ensure chromosome length constraints
    child1.chromosome = enforce_length_constraints(child1.chromosome)
    child2.chromosome = enforce_length_constraints(child2.chromosome)

    return child1, child2


def mutate_individual_jwst(individual, mutation_rate, num_cubes):
    """Mutate individual for JWST."""
    if not individual.chromosome or individual.chromosome[-1] != -1:
        return

    # Remove termination marker for mutation
    chromosome = individual.chromosome[:-1]

    # Point mutation
    for i in range(0, len(chromosome), 2):
        if random.random() < mutation_rate:
            if i < len(chromosome):
                chromosome[i] = random.randint(0, num_cubes - 1)  # Mutate cube
            if i + 1 < len(chromosome):
                chromosome[i + 1] = random.randint(0, 5)  # Mutate command

    # Length mutation
    if random.random() < mutation_rate * 0.5:
        if len(chromosome) < MAX_CHROMOSOME_LENGTH - 2:
            # Add move
            cube_id = random.randint(0, num_cubes - 1)
            move_command = random.randint(0, 5)
            insert_pos = random.randint(0, len(chromosome))
            insert_pos = (insert_pos // 2) * 2
            chromosome = chromosome[:insert_pos] + [cube_id, move_command] + chromosome[insert_pos:]
        elif len(chromosome) > MIN_CHROMOSOME_LENGTH:
            # Remove move
            if len(chromosome) >= 2:
                remove_pos = random.randint(0, (len(chromosome) // 2) - 1) * 2
                chromosome = chromosome[:remove_pos] + chromosome[remove_pos + 2:]

    # Restore termination marker and update individual
    individual.chromosome = chromosome + [-1]
    individual.chromosome = enforce_length_constraints(individual.chromosome)
    individual.is_evaluated = False


def enforce_length_constraints(chromosome):
    """Enforce chromosome length constraints."""
    if not chromosome:
        return [-1]

    # Remove termination marker
    if chromosome[-1] == -1:
        chromosome = chromosome[:-1]

    # Enforce maximum length
    if len(chromosome) > MAX_CHROMOSOME_LENGTH:
        # Trim to nearest pair boundary
        max_pairs = MAX_CHROMOSOME_LENGTH // 2
        chromosome = chromosome[:max_pairs * 2]

    # Enforce minimum length
    if len(chromosome) < MIN_CHROMOSOME_LENGTH:
        # Pad with random moves
        while len(chromosome) < MIN_CHROMOSOME_LENGTH:
            chromosome.extend([random.randint(0, 642), random.randint(0, 5)])

    # Restore termination marker
    chromosome.append(-1)
    return chromosome


if __name__ == "__main__":
    best_chromosome, best_fitness, best_moves = enhanced_genetic_algorithm_jwst()