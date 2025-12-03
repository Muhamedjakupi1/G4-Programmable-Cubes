#!/usr/bin/env python3
"""
Enhanced Genetic Algorithm Submission System for JWST Problem
Academic Research Implementation - GECCO 2024 Space Optimization Competition (SpOC)

ADVANCED ALGORITHMIC FEATURES:
- Population-based evolution with intelligent initialization
- Adaptive mutation mechanisms with convergence detection
- Solution memory banking for pattern learning
- Tournament selection with elitism preservation
- JWST-specific scaling optimization (643 cubes)
- Comprehensive experimental data collection and analysis

This implementation represents advanced genetic algorithm methodology
specifically optimized for JWST (James Webb Space Telescope) spacecraft assembly.

Research Objective: Achieve competitive performance through systematic optimization
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

from solver.optimizers.jwst.enhanced_ga_solver import enhanced_genetic_algorithm_jwst


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
    Execute enhanced genetic algorithm and generate academic submission.

    Returns:
        str: Path to generated submission file
    """
    print("Enhanced Genetic Algorithm for JWST Problem")
    print("Advanced Algorithmic Features:")
    print("  • Population-based evolution with intelligent initialization")
    print("  • Adaptive mutation mechanisms with convergence detection")
    print("  • Solution memory banking for pattern learning")
    print("  • Tournament selection with elitism preservation")
    print("  • JWST-specific scaling optimization")
    print("  • Comprehensive experimental data collection")
    print("Genetic Algorithm Configuration: Population=50, Generations=120")
    print("Research Objective: Competitive JWST spacecraft assembly")
    print("=" * 70)

    start_time = time.time()

    # Execute the enhanced genetic algorithm
    print("Executing enhanced genetic algorithm for JWST...")
    best_chromosome, best_fitness, best_moves = enhanced_genetic_algorithm_jwst()

    execution_time = time.time() - start_time

    print(f"\nEnhanced Genetic Algorithm Completed")
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
    decision_vector = generate_fixed_length_solution_vector(best_chromosome, 30000)  # JWST max_cmds

    # Create submission with standardized format
    print(f"\nGenerating academic submission file...")

    challenge_id = "spoc-3-programmable-cubes"
    problem_id = "jwst"

    output_file = os.path.join(repo_root, "submissions", "enhanced_ga_jwst_submission.json")

    # Ensure submissions directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    submission_name = f"Enhanced Genetic Algorithm JWST v1.0"

    submission_description = (
        f"Enhanced Genetic Algorithm for JWST spacecraft assembly problem. "
        f"Advanced features: (1) Population-based evolution with intelligent initialization, "
        f"(2) Adaptive mutation mechanisms with convergence detection, "
        f"(3) Solution memory banking for pattern learning, "
        f"(4) Tournament selection with elitism preservation, "
        f"(5) JWST-specific scaling optimization for 643 cubes, "
        f"(6) Comprehensive experimental data collection and analysis. "
        f"Academic implementation: Population=50, Generations=120, "
        f"adaptive mechanisms for optimal performance. "
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
    print("Enhanced Genetic Algorithm Submission Generated")
    print(f"Final fitness achieved: {current_fitness:.6f}")
    print(f"Optimization methodology: Enhanced Genetic Algorithm")
    print(f"Total execution time: {execution_time:.1f} seconds")
    print("Advanced features: Population evolution, Memory banking, Adaptive mutation")
    print("=" * 70)

    return output_file


if __name__ == "__main__":
    submission_file = main()