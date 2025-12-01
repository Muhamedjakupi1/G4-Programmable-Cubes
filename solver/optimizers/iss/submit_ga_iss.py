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
