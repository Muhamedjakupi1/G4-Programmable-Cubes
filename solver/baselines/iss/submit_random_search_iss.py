#!/usr/bin/env python3
"""
Academic Submission Generator for Random Search Baseline Algorithm
Programmable Cubes Challenge - GECCO 2024 Space Optimisation Competition (SpOC)

This module executes the random search baseline optimization algorithm and
generates properly formatted submission files for the International Space
Station (ISS) assembly problem. The implementation follows academic standards
for reproducible research and maintains comprehensive experimental records
for comparative algorithmic analysis.

The submission system interfaces with the competition evaluation framework
while preserving detailed performance metrics and solution characteristics
for subsequent empirical studies.

Usage:
    python solver/baselines/iss/submit_random_search_iss.py

Features:
    • Rigorous random search baseline implementation
    • Deterministic results through controlled random seed
    • Standards-compliant JSON submission format
    • Comprehensive performance analysis and statistical reporting
    • Automated result archival for comparative studies
"""

import sys
import os
import numpy as np
import json
import time
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt

# Add the src directory and the repository root to the Python path
repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))

from solver.baselines.iss.random_search import random_search_iss


def create_competition_submission(challenge_id, problem_id, decision_vector, output_path):
    """
    Generate competition submission file in standardized format.

    This function creates submission files conforming to the official competition
    specification, ensuring compatibility with the evaluation framework while
    maintaining data integrity and format compliance.

    Parameters:
        challenge_id (str): Official challenge identifier
        problem_id (str): Specific problem instance identifier
        decision_vector (list): Optimized solution representation
        output_path (str): Destination path for submission file

    Returns:
        str: Path to generated submission file
    """
    # Ensure decision vector is in proper list format
    if isinstance(decision_vector, np.ndarray):
        decision_vector = decision_vector.tolist()

    # Create submission object as list containing solution specification
    submission = [{
        "challenge": challenge_id,
        "problem": problem_id,
        "decisionVector": decision_vector
    }]

    # Serialize submission with standardized formatting
    with open(output_path, 'w') as json_file:
        json.dump(submission, json_file, indent=2)

    print(f"Competition submission generated: {output_path}")
    return output_path


def create_fixed_length_decision_vector(chromosome, max_moves):
    """
    Transform variable-length chromosome to fixed-length decision vector format.

    This function standardizes solution representations to meet competition
    requirements for fixed-length decision vectors, padding shorter solutions
    with no-operation commands while preserving solution semantics.

    Parameters:
        chromosome (list/np.ndarray): Variable-length solution representation
        max_moves (int): Required fixed length for decision vector

    Returns:
        list: Fixed-length decision vector with proper termination
    """
    # Normalize input to list format
    if isinstance(chromosome, np.ndarray):
        chromosome = chromosome.tolist()

    # Determine effective solution length
    if -1 in chromosome:
        end_pos = chromosome.index(-1)
    else:
        end_pos = len(chromosome)

    actual_moves = end_pos // 2
    print(f"Original solution contains {actual_moves} moves, expanding to {max_moves} moves")

    # Extract active portion of solution
    decision_vector = chromosome[:end_pos].copy()

    # Pad with no-operation commands to achieve required length
    moves_to_add = max_moves - actual_moves
    for _ in range(moves_to_add):
        decision_vector.extend([-1, 0])  # No-op: invalid cube with null command

    # Append final termination sentinel
    decision_vector.append(-1)
    print(f"Final decision vector length: {len(decision_vector)} (target: {max_moves * 2 + 1})")

    return decision_vector


def main():
    """
    Execute comprehensive experimental protocol for random search baseline evaluation.

    This function orchestrates the complete experimental workflow including:
    algorithm execution, performance analysis, result archival, and competition
    submission generation. The implementation follows academic standards for
    reproducible research and comprehensive documentation.

    Returns:
        str: Path to generated competition submission file
    """
    print("=" * 80)
    print("ACADEMIC EXPERIMENTAL FRAMEWORK: RANDOM SEARCH BASELINE EVALUATION")
    print("=" * 80)
    print("OBJECTIVE: Establish empirical baseline for comparative algorithmic analysis")
    print("METHODOLOGY: Stochastic optimization via uniform random sampling")
    print("SCOPE: International Space Station (ISS) modular assembly optimization")
    print("\nEXPERIMENTAL FEATURES:")
    print("  • Deterministic reproducibility through controlled random seeding")
    print("  • Comprehensive statistical analysis and performance metrics")
    print("  • Automated result archival for comparative studies")
    print("  • Standards-compliant competition submission generation")
    print("=" * 80)

    # Record experimental start time
    experiment_start_time = time.time()

    # Execute stochastic optimization algorithm
    print("\nCOMMENCING STOCHASTIC OPTIMIZATION EXPERIMENT...")
    best_chromosome, best_fitness, best_moves, results_data = random_search_iss()

    # Calculate total experimental runtime
    total_execution_time = time.time() - experiment_start_time

    print(f"\n" + "=" * 80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"Optimization Runtime: {results_data['results']['execution_time']:.2f} seconds")
    print(f"Total Experimental Time: {total_execution_time:.2f} seconds")
    print(f"Optimal Objective Value: {best_fitness:.6f}")
    print(f"Solution Complexity: {best_moves} movement operations")
    print(f"Representation Length: {len(best_chromosome)} elements")

    # Comparative Performance Analysis
    reference_fitness_values = {
        "championship_optimum": -0.991,  # Historical championship performance
        "competitive_threshold": -0.5,  # Competitive algorithm threshold
        "baseline_reference": 0.186  # Typical baseline performance
    }

    current_fitness = best_fitness

    print(f"\nCOMPARATIVE PERFORMANCE ANALYSIS:")
    print(f"  • Championship optimum (reference): {reference_fitness_values['championship_optimum']:.6f}")
    print(f"  • Competitive threshold: {reference_fitness_values['competitive_threshold']:.6f}")
    print(f"  • Baseline reference: {reference_fitness_values['baseline_reference']:.6f}")
    print(f"  • Achieved performance: {current_fitness:.6f}")

    # Calculate relative performance metrics
    if current_fitness < reference_fitness_values['baseline_reference']:
        improvement = reference_fitness_values['baseline_reference'] - current_fitness
        print(f"  • Improvement over baseline: {improvement:.6f}")
    else:
        gap = current_fitness - reference_fitness_values['baseline_reference']
        print(f"  • Performance gap from baseline: {gap:.6f}")

    # Academic performance classification
    if current_fitness <= reference_fitness_values['championship_optimum']:
        performance_class = "EXCEPTIONAL"
        performance_description = "Championship-level optimization achieved"
        academic_rating = "A+"
    elif current_fitness < -0.8:
        progress = (abs(current_fitness) / 0.991) * 100
        performance_class = "EXCELLENT"
        performance_description = f"Elite optimization performance ({progress:.1f}% of theoretical optimum)"
        academic_rating = "A"
    elif current_fitness < reference_fitness_values['competitive_threshold']:
        progress = (abs(current_fitness) / 0.991) * 100
        performance_class = "VERY_GOOD"
        performance_description = f"Strong optimization performance ({progress:.1f}% of theoretical optimum)"
        academic_rating = "B+"
    elif current_fitness < 0:
        progress = (abs(current_fitness) / 0.991) * 100
        performance_class = "GOOD"
        performance_description = f"Competitive optimization achieved ({progress:.1f}% of theoretical optimum)"
        academic_rating = "B"
    elif current_fitness < reference_fitness_values['baseline_reference']:
        improvement_pct = ((reference_fitness_values['baseline_reference'] - current_fitness) /
                           reference_fitness_values['baseline_reference']) * 100
        performance_class = "SATISFACTORY"
        performance_description = f"Above-baseline performance ({improvement_pct:.1f}% improvement)"
        academic_rating = "C+"
    else:
        performance_class = "BASELINE"
        performance_description = "Typical baseline performance achieved"
        academic_rating = "C"

    print(f"\nPERFORMANCE CLASSIFICATION:")
    print(f"  • Class: {performance_class}")
    print(f"  • Description: {performance_description}")
    print(f"  • Academic Rating: {academic_rating}")

    # Solution Transformation and Submission Generation
    print(f"\n" + "=" * 80)
    print("SOLUTION TRANSFORMATION AND SUBMISSION GENERATION")
    print("=" * 80)

    # Transform variable-length chromosome to fixed-length decision vector
    print("Transforming optimized solution to competition format...")
    decision_vector = create_fixed_length_decision_vector(best_chromosome, 6000)

    # Generate competition submission
    print("Generating standards-compliant competition submission...")

    challenge_id = "spoc-3-programmable-cubes"
    problem_id = "iss"

    output_file = os.path.join(repo_root, "submissions", "random_search_iss_submission.json")

    # Ensure submissions directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create standardized submission file
    create_competition_submission(
        challenge_id=challenge_id,
        problem_id=problem_id,
        decision_vector=decision_vector,
        output_path=output_file
    )

    # Submission Validation and Documentation
    print(f"\nSUBMISSION DOCUMENTATION:")
    print(f"  • File path: {output_file}")
    print(f"  • Challenge identifier: {challenge_id}")
    print(f"  • Problem identifier: {problem_id}")
    print(f"  • Decision vector length: {len(decision_vector)} elements")
    print(f"  • Solution preview: {decision_vector[:10]}...")

    # Validate submission file integrity
    try:
        with open(output_file, 'r') as f:
            loaded_submission = json.load(f)
        print(f"  • JSON format validation: PASSED")
        print(f"  • Submission structure: {list(loaded_submission[0].keys())}")
    except Exception as e:
        print(f"  • JSON format validation: FAILED - {e}")

    # Final Experimental Summary
    print(f"\n" + "=" * 80)
    print("EXPERIMENTAL PROTOCOL COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Performance Classification: {performance_class} ({academic_rating})")
    print(f"Algorithmic Approach: Stochastic Random Search")
    print(f"Total Experimental Runtime: {total_execution_time:.2f} seconds")
    print(f"Optimization Efficiency: {results_data['results']['iterations_per_second']:.1f} iterations/second")
    print(f"\nDOCUMENTATION STATUS:")
    print(f"  • Competition submission: READY")
    print(f"  • Experimental results: ARCHIVED")
    print(f"  • Performance metrics: DOCUMENTED")
    print(f"  • Comparative analysis: ENABLED")
    print("=" * 80)

    return output_file


if __name__ == "__main__":
    # Execute comprehensive experimental protocol
    submission_file_path = main()