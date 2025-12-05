#!/usr/bin/env python3
"""
Submit Greedy Solver Solution for Enterprise Problem
Programmable Cubes Challenge - GECCO 2024 Space Optimisation Competition (SpOC)

This script runs the greedy solver for the Enterprise problem and creates a submission file.
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

from solver.heuristics.enterprise.greedy_solver import greedy_heuristic_optimization_enterprise


def create_correct_submission(challenge_id, problem_id, decision_vector, fn_out):
    """
    Create submission file in the CORRECT format as specified in the competition guidelines.

    Format should be a LIST of solution objects:
    [{
      "challenge": "challenge_id",
      "problem": "problem_id",
      "decisionVector": [decision_vector]
    }]
    """
    # Convert numpy arrays to Python lists
    if isinstance(decision_vector, np.ndarray):
        decision_vector = decision_vector.tolist()

    # Create the submission as a LIST of solution objects
    submission = [{
        "challenge": challenge_id,
        "problem": problem_id,
        "decisionVector": decision_vector
    }]

    # Write to file with proper formatting
    with open(fn_out, 'w') as json_file:
        json.dump(submission, json_file, indent=2)

    print(f"Correct format submission created: {fn_out}")
    return fn_out


def create_fixed_length_decision_vector(chromosome, max_moves):
    """
    Convert variable-length chromosome to fixed-length decision vector.

    Args:
        chromosome (np.ndarray): Variable-length chromosome ending with -1
        max_moves (int): Maximum number of moves (100000 for Enterprise)

    Returns:
        list: Fixed-length decision vector of length (max_moves * 2 + 1)
    """
    # Convert numpy array to list if needed
    if isinstance(chromosome, np.ndarray):
        chromosome = chromosome.tolist()

    if -1 in chromosome:
        end_pos = chromosome.index(-1)
    else:
        end_pos = len(chromosome)

    actual_moves = end_pos // 2
    print(f"Original chromosome has {actual_moves} moves, expanding to {max_moves} moves")

    decision_vector = chromosome[:end_pos].copy()

    # Pad with no-op moves
    moves_to_add = max_moves - actual_moves
    for _ in range(moves_to_add):
        decision_vector.extend([-1, 0])

    decision_vector.append(-1)
    print(f"Final decision vector length: {len(decision_vector)} (should be {max_moves * 2 + 1})")

    return decision_vector


def main():
    """
    Run the greedy heuristic optimization and create a submission file for the Enterprise problem.
    """
    print("LAUNCHING GREEDY HEURISTIC OPTIMIZATION")
    print("TARGET: Establish competitive performance for Enterprise problem")
    print("FEATURES:")
    print("   • Target-aware cube selection with spatial reasoning")
    print("   • Adaptive greedy/exploration balance with temperature control")
    print("   • Enhanced move evaluation using multiple heuristics")
    print("   • Progressive penalty system for move redundancy prevention")
    print("   • Proper JSON submission format")
    print("=" * 70)

    start_time = time.time()

    # Run the greedy heuristic optimization
    print("Running greedy heuristic optimization for Enterprise problem...")
    optimal_chromosome, optimal_fitness, optimal_moves, experimental_results = greedy_heuristic_optimization_enterprise()

    execution_time = time.time() - start_time

    print(f"\nGREEDY HEURISTIC OPTIMIZATION COMPLETED!")
    print(f"Execution time: {execution_time:.1f} seconds")
    print(f"Best fitness: {optimal_fitness:.6f}")
    print(f"Optimal moves: {optimal_moves}")
    print(f"Configuration length: {len(optimal_chromosome)}")

    # Performance analysis for Enterprise
    target_fitness = -0.8  # Championship target for Enterprise
    baseline_fitness = 0.015  # Random search baseline for Enterprise
    current_fitness = optimal_fitness

    print(f"\nPERFORMANCE ANALYSIS:")
    print(f"Random search baseline: {baseline_fitness:.6f}")
    print(f"Target performance (1st place): {target_fitness:.6f}")
    print(f"Achieved performance: {current_fitness:.6f}")

    # Calculate improvement over random search
    if current_fitness < baseline_fitness:
        improvement = baseline_fitness - current_fitness
        improvement_pct = (improvement / baseline_fitness) * 100
        print(f"Improvement over random search: {improvement:.6f} ({improvement_pct:.1f}%)")
    else:
        gap = current_fitness - baseline_fitness
        print(f"Gap from random search baseline: {gap:.6f}")

    print(f"Performance status: Academic research implementation")
    print(f"Algorithm type: Greedy heuristic optimization")

    # Convert chromosome to decision vector
    print(f"\nConverting chromosome to decision vector...")
    decision_vector = create_fixed_length_decision_vector(optimal_chromosome, 100000)

    # Create submission with CORRECT format
    print(f"\nCreating correctly formatted submission...")

    challenge_id = "spoc-3-programmable-cubes"
    problem_id = "enterprise"

    output_file = os.path.join(repo_root, "submissions", "greedy_enterprise_submission.json")

    # Ensure submissions directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create the submission with correct format
    create_correct_submission(
        challenge_id=challenge_id,
        problem_id=problem_id,
        decision_vector=decision_vector,
        fn_out=output_file
    )

    print(f"\nSUBMISSION DETAILS:")
    print(f"File: {output_file}")
    print(f"Challenge ID: {challenge_id}")
    print(f"Problem ID: {problem_id}")
    print(f"Decision vector length: {len(decision_vector)}")
    print(f"Preview: {decision_vector[:10]}...")

    # Validate JSON format
    try:
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)
        print(f"JSON format validation: PASSED")
        print(f"Submission structure: {type(loaded_data)} with {len(loaded_data)} solution(s)")
        if len(loaded_data) > 0:
            print(f"Keys in first solution: {list(loaded_data[0].keys())}")
    except Exception as e:
        print(f"JSON format validation: FAILED - {e}")

    print(f"\n" + "=" * 70)
    print("GREEDY HEURISTIC OPTIMIZATION SUBMISSION READY!")
    print(f"Performance Level: Academic research implementation")
    print(f"Algorithm Type: Greedy heuristic optimization")
    print(f"Total time: {execution_time:.1f} seconds")
    print("=" * 70)

    return output_file


if __name__ == "__main__":
    submission_file = main()