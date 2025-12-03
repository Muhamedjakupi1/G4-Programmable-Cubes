#!/usr/bin/env python3
"""
Academic Submission Generator for JWST Stochastic Optimization Research
GECCO 2024 Space Optimization Competition (SpOC) - Academic Implementation

This module generates properly formatted competition submissions based on
stochastic optimization baseline results for the James Webb Space Telescope
(JWST) spacecraft assembly problem. The implementation facilitates rigorous
academic research by providing comprehensive performance analysis and
standardized result documentation for comparative algorithmic studies.

Academic Usage:
    python solver/baselines/jwst/submit_random_search_jwst.py

Research Features:
    • Rigorous stochastic optimization implementation with academic terminology
    • Reproducible experimental results with comprehensive documentation
    • Performance classification system for competitive analysis
    • Structured JSON submission format compliant with competition standards
    • Automated result storage for comparative research studies
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

from solver.baselines.jwst.random_search import stochastic_search_jwst


def create_competition_submission(challenge_id, problem_id, decision_vector, fn_out):
    """
    Create properly formatted competition submission for academic research.

    This function generates standardized JSON submission files compliant with
    competition specifications and suitable for academic documentation.

    Format specification: List of solution objects containing challenge metadata
    and decision vector encoding for algorithmic performance evaluation.

    Args:
        challenge_id (str): Competition challenge identifier
        problem_id (str): Specific problem instance identifier
        decision_vector (list): Optimized solution encoding
        fn_out (str): Output file path for submission

    Returns:
        str: Path to generated submission file
    """
    # Convert numpy arrays to Python lists for JSON serialization
    if isinstance(decision_vector, np.ndarray):
        decision_vector = decision_vector.tolist()

    # Create submission using specified format: list of solution objects
    submission = [{
        "challenge": challenge_id,
        "problem": problem_id,
        "decisionVector": decision_vector
    }]

    # Write formatted submission with proper indentation
    with open(fn_out, 'w') as json_file:
        json.dump(submission, json_file, indent=2)

    print(f"Competition submission generated: {fn_out}")
    return fn_out


def generate_standardized_decision_vector(chromosome, max_moves):
    """
    Generate standardized fixed-length decision vector for competition submission.

    This function converts variable-length chromosome encodings to standardized
    fixed-length decision vectors compliant with competition requirements.

    Args:
        chromosome: Variable-length solution chromosome
        max_moves (int): Maximum number of movement operations allowed

    Returns:
        list: Fixed-length decision vector with padding operations
    """
    # Convert numpy array to list if needed
    if isinstance(chromosome, np.ndarray):
        chromosome = chromosome.tolist()

    # Identify actual solution length (excluding sentinel)
    if -1 in chromosome:
        end_pos = chromosome.index(-1)
    else:
        end_pos = len(chromosome)

    actual_moves = end_pos // 2
    print(f"Original solution complexity: {actual_moves} operations, standardizing to {max_moves} operations")

    # Extract actual solution sequence
    decision_vector = chromosome[:end_pos].copy()

    # Pad with no-operation sequences to reach standard length
    moves_to_add = max_moves - actual_moves
    for _ in range(moves_to_add):
        decision_vector.extend([-1, 0])  # No-op padding

    # Add termination sentinel
    decision_vector.append(-1)
    print(f"Standardized decision vector length: {len(decision_vector)} (expected: {max_moves * 2 + 1})")

    return decision_vector


def main():
    """Execute stochastic optimization and generate academic competition submission."""
    print("INITIATING STOCHASTIC OPTIMIZATION RESEARCH")
    print("OBJECTIVE: Establish academic baseline for JWST spacecraft assembly")
    print("ACADEMIC FEATURES:")
    print("   • Rigorous stochastic optimization with Monte Carlo sampling")
    print("   • Reproducible experimental methodology with fixed random seed")
    print("   • Comprehensive performance analysis and documentation")
    print("   • Standardized competition submission format")
    print("   • Academic result storage for comparative studies")
    print("=" * 80)

    start_time = time.time()

    # Execute stochastic optimization algorithm
    print("Executing stochastic optimization algorithm...")
    best_chromosome, best_fitness, best_moves, fitness_history = stochastic_search_jwst()

    execution_time = time.time() - start_time

    print(f"\nSTOCHASTIC OPTIMIZATION COMPLETED")
    print(f"Execution time: {execution_time:.1f} seconds")
    print(f"Optimal fitness: {best_fitness:.6f}")
    print(f"Solution complexity: {best_moves} operations")
    print(f"Chromosome encoding length: {len(best_chromosome)}")

    # Academic performance documentation
    print(f"\nACCADEMIC PERFORMANCE ANALYSIS:")
    print(f"Achieved experimental result: {best_fitness:.6f}")
    print(f"Solution complexity: {best_moves} movement operations")
    print(f"Algorithm execution time: {execution_time:.2f} seconds")

    # Generate standardized decision vector for submission
    print(f"\nGenerating standardized decision vector for competition submission...")
    decision_vector = generate_standardized_decision_vector(best_chromosome, 30000)

    # Create academic competition submission
    print(f"\nCreating academic competition submission...")

    challenge_id = "spoc-3-programmable-cubes"
    problem_id = "jwst"

    output_file = os.path.join(repo_root, "submissions", "random_search_jwst_submission.json")

    # Ensure submissions directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Generate the competition submission
    create_competition_submission(
        challenge_id=challenge_id,
        problem_id=problem_id,
        decision_vector=decision_vector,
        fn_out=output_file
    )

    print(f"\nACADEMIC SUBMISSION DETAILS:")
    print(f"Submission file: {output_file}")
    print(f"Challenge identifier: {challenge_id}")
    print(f"Problem identifier: {problem_id}")
    print(f"Decision vector length: {len(decision_vector)}")
    print(f"Decision vector preview: {decision_vector[:10]}...")

    # Validate JSON format integrity
    try:
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)
        print(f"JSON format validation: SUCCESSFUL")
        if isinstance(loaded_data, list) and len(loaded_data) > 0:
            print(f"Submission structure: {list(loaded_data[0].keys())}")
        else:
            print(f"Submission structure: {type(loaded_data)}")
    except Exception as e:
        print(f"JSON format validation: FAILED - {e}")

    print(f"\n" + "=" * 80)
    print("ACADEMIC STOCHASTIC OPTIMIZATION RESEARCH COMPLETED")
    print(f"Total research time: {execution_time:.1f} seconds")
    print("Results available for comparative algorithmic studies")
    print("=" * 80)

    return output_file


if __name__ == "__main__":
    submission_file = main()