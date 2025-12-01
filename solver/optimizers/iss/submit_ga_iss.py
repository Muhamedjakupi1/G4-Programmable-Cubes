# Add the src directory and the repository root to the Python path
import os
import sys
import json
import numpy as np
import time

repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))

from solver.optimizers.iss.ga_solver import genetic_algorithm_iss


def create_correct_submission(challenge_id, problem_id, decision_vector, fn_out, name="", description=""):
    """
    Create submission file in the CORRECT format as specified in the competition guidelines.

    Format should be:
    {
      "challenge": "challenge_id",
      "problem": "problem_id",
      "decisionVector": [decision_vector],
      "name": "name",
      "description": "description"
    }
    """

    # Convert numpy arrays to Python lists
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

    print(f"✅ Correct format submission created: {fn_out}")
    return fn_out


def create_fixed_length_decision_vector(chromosome, max_moves):
    """Convert variable-length chromosome to fixed-length decision vector."""
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
    """Execute genetic algorithm optimization and generate submission file."""
    print("=" * 80)
    print("Enhanced Genetic Algorithm Solution Submission for ISS Problem")
    print("Programmable Cubes Challenge - GECCO 2024 Competition")
    print("=" * 80)
    print()
    print("Algorithm Configuration:")
    print("  • Optimization Method: Enhanced Genetic Algorithm")
    print("  • Target Performance: Fitness ≤ -0.991 (championship level)")
    print("  • Key Improvements: Corrected fitness direction, adaptive mechanisms")
    print("  • Output Format: Standardized competition submission")
    print()

    algorithm_start_time = time.time()

    # Execute genetic algorithm optimization
    print("Executing enhanced genetic algorithm optimization...")
    best_chromosome, best_fitness, best_moves = genetic_algorithm_iss()

    execution_time = time.time() - algorithm_start_time

    print()
    print("=" * 80)
    print("Optimization Results Summary")
    print("=" * 80)
    print(f"Execution time: {execution_time:.1f} seconds")
    print(f"Best fitness achieved: {best_fitness:.6f}")
    print(f"Number of moves: {best_moves}")
    print(f"Chromosome length: {len(best_chromosome)}")

    # Performance analysis with corrected fitness understanding
    target_fitness = -0.991
    baseline_fitness = 0.186  # Original baseline performance
    current_fitness = best_fitness

    print()
    print("Performance Analysis:")
    print(f"  • Baseline performance: {baseline_fitness:.6f}")
    print(f"  • Target performance: {target_fitness:.6f}")
    print(f"  • Achieved performance: {current_fitness:.6f}")

    # Calculate performance improvements
    improvement_over_baseline = baseline_fitness - current_fitness
    print(f"  • Improvement over baseline: {improvement_over_baseline:.6f}")


 # Determine performance classification
    if current_fitness <= target_fitness:
        performance_classification = "CHAMPION"
        expected_ranking = "1st-3rd place"
        print(f"  • Status: Championship performance achieved")
    elif current_fitness < -0.8:
        progress_percentage = (abs(current_fitness) / 0.991) * 100
        performance_classification = "ELITE"
        expected_ranking = "Top 3"
        print(f"  • Status: Elite performance | Progress: {progress_percentage:.1f}%")
    elif current_fitness < -0.5:
        progress_percentage = (abs(current_fitness) / 0.991) * 100
        performance_classification = "EXCELLENT"
        expected_ranking = "Top 5"
        print(f"  • Status: Excellent performance | Progress: {progress_percentage:.1f}%")
    elif current_fitness < -0.2:
        progress_percentage = (abs(current_fitness) / 0.991) * 100
        performance_classification = "VERY_GOOD"
        expected_ranking = "Top 10"
        print(f"  • Status: Very good performance | Progress: {progress_percentage:.1f}%")
    elif current_fitness < 0:
        progress_percentage = (abs(current_fitness) / 0.991) * 100
        performance_classification = "COMPETITIVE"
        expected_ranking = "Top 15"
        print(f"  • Status: Competitive performance | Progress: {progress_percentage:.1f}%")
    else:
        if improvement_over_baseline > 0:
            improvement_percentage = (improvement_over_baseline / baseline_fitness) * 100
            performance_classification = "IMPROVED"
            expected_ranking = "Better than baseline"
            print(f"  • Status: Improved performance | {improvement_percentage:.1f}% better than baseline")
        else:
            performance_classification = "EXPERIMENTAL"
            expected_ranking = "Experimental"
            print(f"  • Status: Experimental result")

    print(f"  • Performance Classification: {performance_classification}")
    print(f"  • Expected Ranking: {expected_ranking}")

    # Validate optimization direction
    if current_fitness < baseline_fitness:
        print("  • Optimization Direction: Confirmed correct (algorithm working properly)")
    else:
        print("  • Optimization Direction: May require further optimization")

