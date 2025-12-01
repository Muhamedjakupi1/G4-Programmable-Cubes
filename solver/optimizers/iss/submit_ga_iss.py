# Add the src directory and the repository root to the Python path
import os
import sys
import json
import numpy as np

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
