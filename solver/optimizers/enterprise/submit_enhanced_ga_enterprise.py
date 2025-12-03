#!/usr/bin/env python3
"""
Enhanced Multi-Population Genetic Algorithm Submission System for Enterprise Problem
Academic Research Implementation - GECCO 2024 Space Optimization Competition (SpOC)

ADVANCED ALGORITHMIC FEATURES:
- Multi-population evolution with structured migration
- Solution memory banking and pattern learning
- Novelty-driven diversity preservation mechanisms
- Advanced crossover and mutation operators
- Enterprise-specific scaling optimization (1472 cubes)
- Adaptive parameter control systems
- Tabu-guided search enhancement

This implementation represents advanced genetic algorithm methodology
specifically optimized for large-scale Enterprise spacecraft assembly.

Research Objective: Achieve competitive performance through algorithmic innovation
"""

import sys
import os
import numpy as np
import json
import time

# Add the src directory and the repository root to the Python path
repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))

from solver.optimizers.enterprise.enhanced_ga_solver import enhanced_genetic_algorithm_enterprise


def create_submission_file(challenge_id, problem_id, decision_vector, fn_out, name="", description=""):
    """
    Generate submission file in standardized competition format.

    Args:
        challenge_id: Competition challenge identifier
        problem_id: Specific problem instance identifier
        decision_vector: Optimized solution vector
        fn_out: Output file path for submission
        name: Submission identifier
        description: Technical description of methodology

    Returns:
        str: Path to generated submission file
    """
    # Convert numpy arrays to Python lists for JSON serialization
    if isinstance(decision_vector, np.ndarray):
        decision_vector = decision_vector.tolist()

    # Create the submission object in the correct format
    submission = {
        "challenge": challenge_id,
        "problem": problem_id,
        "decisionVector": decision_vector,
        "name": name,
        "description": description
    }

    # Write to file with proper formatting
    with open(fn_out, 'w') as json_file:
        json.dump(submission, json_file, indent=2)

    print(f"Academic submission file created: {fn_out}")
    return fn_out


def generate_fixed_length_solution_vector(chromosome, max_moves):
    """
    Convert variable-length genetic algorithm chromosome to fixed-length decision vector.

    Args:
        chromosome: Variable-length chromosome from genetic algorithm
        max_moves: Maximum number of moves allowed by problem constraints

    Returns:
        list: Fixed-length decision vector padded with no-operation moves
    """
    if -1 in chromosome:
        end_pos = chromosome.index(-1)
    else:
        end_pos = len(chromosome)

    actual_moves = end_pos // 2
    print(f"Original chromosome contains {actual_moves} moves, expanding to {max_moves} maximum moves")

    decision_vector = chromosome[:end_pos].copy()

    # Pad with no-operation moves to reach maximum length
    moves_to_add = max_moves - actual_moves
    for _ in range(moves_to_add):
        decision_vector.extend([-1, 0])

    decision_vector.append(-1)
    print(f"Generated decision vector length: {len(decision_vector)} (expected: {max_moves * 2 + 1})")

    return decision_vector


def main():
    """
    Execute enhanced multi-population genetic algorithm and generate academic submission.

    Returns:
        str: Path to generated submission file
    """
    print("Enhanced Multi-Population Genetic Algorithm for Enterprise Problem")
    print("Advanced Algorithmic Features:")
    print("  • Multi-population evolution with structured migration")
    print("  • Solution memory banking and pattern learning")
    print("  • Novelty-driven diversity preservation mechanisms")
    print("  • Advanced crossover and mutation operators")
    print("  • Enterprise-specific scaling optimization")
    print("  • Adaptive parameter control systems")
    print("  • Tabu-guided search enhancement")
    print("Genetic Algorithm Configuration: Population=60, Generations=150")
    print("Research Objective: Competitive Enterprise spacecraft assembly")
    print("=" * 70)

    start_time = time.time()

    # Execute the enhanced genetic algorithm
    print("Executing enhanced multi-population genetic algorithm for Enterprise...")
    best_chromosome, best_fitness, best_moves = enhanced_genetic_algorithm_enterprise()

    execution_time = time.time() - start_time

    print(f"\nEnhanced Multi-Population Genetic Algorithm Completed")
    print(f"Execution time: {execution_time:.1f} seconds")
    print(f"Best fitness achieved: {best_fitness:.6f}")
    print(f"Optimal move count: {best_moves}")
    print(f"Chromosome length: {len(best_chromosome)}")

    # Academic performance analysis
    current_fitness = best_fitness

    print(f"\nPerformance Analysis:")
    print(f"Final fitness value: {current_fitness:.6f}")
    print(f"Optimization completed successfully")

    # Convert chromosome to decision vector
    print(f"\nConverting optimized chromosome to standardized decision vector...")
    decision_vector = generate_fixed_length_solution_vector(best_chromosome, 100000)  # Enterprise max_cmds

    # Create submission with standardized format
    print(f"\nGenerating academic submission file...")

    challenge_id = "spoc-3-programmable-cubes"
    problem_id = "enterprise"

    output_file = os.path.join(repo_root, "submissions", "enhanced_ga_enterprise_submission.json")

    # Ensure submissions directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    submission_name = f"Enhanced Multi-Population GA Enterprise v2.0"

    submission_description = (
        f"Enhanced Multi-Population Genetic Algorithm for Enterprise spacecraft assembly problem. "
        f"Advanced features: (1) Multi-population evolution with structured migration, "
        f"(2) Solution memory banking and pattern learning mechanisms, "
        f"(3) Novelty-driven diversity preservation, "
        f"(4) Advanced crossover and mutation operators, "
        f"(5) Enterprise-specific scaling optimization for 1472 cubes, "
        f"(6) Adaptive parameter control and tabu-guided search enhancement. "
        f"Academic implementation: Population=60, Generations=150, "
        f"advanced diversity mechanisms. "
        f"Performance: {current_fitness:.6f} fitness with {best_moves} moves. "
        f"Research-grade optimization methodology."
    )

    # Create the submission with standardized format
    create_submission_file(
        challenge_id=challenge_id,
        problem_id=problem_id,
        decision_vector=decision_vector,
        fn_out=output_file,
        name=submission_name,
        description=submission_description
    )
    print(f"\nAcademic Submission Details:")
    print(f"  • Submission file: {output_file}")
    print(f"  • Challenge ID: {challenge_id}")
    print(f"  • Problem ID: {problem_id}")
    print(f"  • Decision vector length: {len(decision_vector)}")
    print(f"  • Solution preview: {decision_vector[:10]}...")

    # Validate JSON format
    try:
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)
        print(f"JSON format validation: PASSED")
        print(f"Submission keys: {list(loaded_data.keys())}")
    except Exception as e:
        print(f"JSON format validation: FAILED - {e}")

    print(f"\n" + "=" * 70)
    print("Enhanced Multi-Population Genetic Algorithm Submission Generated")
    print(f"Final fitness achieved: {current_fitness:.6f}")

    # Remove performance classification - keeping it neutral
    print(f"Optimization methodology: Enhanced Multi-Population Genetic Algorithm")

    print(f"Total execution time: {execution_time:.1f} seconds")
    print("Advanced features: Multi-population, Memory, Novelty, Tabu")
    print("=" * 70)

    return output_file


if __name__ == "__main__":
    submission_file = main()