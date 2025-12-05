#!/usr/bin/env python3
"""
Academic Submission Generator for Random Search Baseline Algorithm - Enterprise Configuration
Programmable Cubes Challenge - GECCO 2024 Space Optimisation Competition (SpOC)

This module executes the random search baseline optimization algorithm and
generates properly formatted submission files for the Enterprise spacecraft
assembly problem. The implementation follows academic standards for reproducible
research and maintains comprehensive experimental records for comparative
algorithmic analysis.

The Enterprise problem represents the most complex configuration challenge in the
competition, featuring 1472 programmable cubes requiring sophisticated optimization
strategies for effective structural assembly.

The submission system interfaces with the competition evaluation framework
while preserving detailed performance metrics and solution characteristics
for subsequent empirical studies.

Usage:
    python solver/baselines/enterprise/submit_random_search_enterprise.py

Features:
    • Rigorous random search baseline implementation for large-scale problems
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

from solver.baselines.enterprise.random_search import random_search_enterprise


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
