#!/usr/bin/env python3
"""
Academic Submission Generator for ISS Greedy Heuristic Optimization Research
GECCO 2024 Space Optimization Competition (SpOC) - Academic Implementation

This module generates properly formatted competition submissions based on 
greedy heuristic optimization results for the International Space Station (ISS)
spacecraft assembly problem. The implementation facilitates rigorous academic
research by providing comprehensive performance analysis and standardized
result documentation for comparative algorithmic studies.

Academic Usage:
    python solver/heuristics/iss/submit_greedy_iss.py

Research Features:
    • Advanced greedy heuristic optimization with probabilistic exploration
    • Reproducible experimental results with comprehensive documentation
    • Performance enhancement analysis compared to baseline methods
    • Structured JSON submission format compliant with competition standards
    • Automated result storage for comparative research studies
"""

import sys
import os
import numpy as np

# Add the src directory and the repository root to the Python path
repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))

from greedy_solver import greedy_heuristic_optimization_iss
from submission_helper import create_submission

def generate_standardized_decision_vector(chromosome, max_moves):
    """
    Generate standardized fixed-length decision vector for competition submission.
    
    This function converts variable-length chromosome encodings to standardized
    fixed-length decision vectors compliant with competition requirements.
    
    Args:
        chromosome (np.ndarray): Variable-length chromosome ending with -1 sentinel
        max_moves (int): Maximum number of movement operations allowed (6000 for ISS)
        
    Returns:
        list: Fixed-length decision vector of length (max_moves * 2 + 1)
    """
    # Locate the sentinel terminator position
    end_pos = np.where(chromosome == -1)[0][0]
    actual_moves = end_pos // 2  # Number of actual movement operations
    
    print(f"Original solution complexity: {actual_moves} operations, standardizing to {max_moves} operations")
    
    # Extract actual solution sequence
    decision_vector = chromosome[:end_pos].tolist()
    
    # Pad with no-operation sequences to reach standard length
    # Use cube_id=0, move_id=0 as no-operation padding
    moves_to_add = max_moves - actual_moves
    for _ in range(moves_to_add):
        decision_vector.extend([0, 0])  # No-operation move: cube 0, move 0
    
    # Add termination sentinel
    decision_vector.append(-1)
    
    print(f"Standardized decision vector length: {len(decision_vector)} (expected: {max_moves * 2 + 1})")
    
    return decision_vector

def main():
    """
    Execute greedy heuristic optimization and generate academic competition submission.
    
    This function runs the advanced greedy heuristic algorithm and creates a properly
    formatted submission file for academic research and competition evaluation.
    """
    print("INITIATING GREEDY HEURISTIC OPTIMIZATION RESEARCH")
    print("OBJECTIVE: Advanced heuristic optimization for ISS spacecraft assembly")
    print("ACADEMIC FEATURES:")
    print("   • Intelligent greedy construction with probabilistic exploration")
    print("   • Recent movement tracking for redundancy prevention")
    print("   • Reproducible experimental methodology")
    print("   • Performance enhancement analysis")
    print("   • Academic result documentation for comparative studies")
    print("=" * 80)
    
    # Execute greedy heuristic optimization algorithm
    print("Executing greedy heuristic optimization algorithm...")
    best_chromosome, best_fitness, best_moves, fitness_history = greedy_heuristic_optimization_iss()
    
    print(f"\nGREEDY HEURISTIC OPTIMIZATION COMPLETED")
    print(f"Optimal fitness: {best_fitness:.6f}")
    print(f"Solution complexity: {best_moves} operations")
    print(f"Chromosome encoding length: {len(best_chromosome)}")
    
    # Academic performance analysis
    baseline_fitness = 0.043  # Random search baseline reference
    if baseline_fitness > 0:
        improvement = ((best_fitness - baseline_fitness) / baseline_fitness) * 100
        print(f"Performance enhancement over baseline: {improvement:.1f}%")
    
    # Generate standardized decision vector for submission
    print(f"\nGenerating standardized decision vector for competition submission...")
    # ISS requires exactly 12001 elements: 6000 moves (cube_id, move_id pairs) + terminating -1
    decision_vector = generate_standardized_decision_vector(best_chromosome, 6000)
    
    # Create academic competition submission
    print(f"\nCreating academic competition submission...")
    
    # Competition identifiers based on academic standards
    challenge_id = "spoc-3-programmable-cubes"  # Space Optimization Competition
    problem_id = "iss"                          # ISS problem identifier (lowercase)
    
    output_file = os.path.join(repo_root, "submissions", "greedy_iss_submission.json")
    
    # Ensure submissions directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Generate submission using academic description
    create_submission(
        challenge_id=challenge_id,
        problem_id=problem_id,
        x=decision_vector,
        fn_out=output_file,
        name="Greedy Heuristic Optimization - ISS",
        description=f"Advanced greedy heuristic optimization for ISS spacecraft assembly problem. "
                   f"Achieved fitness: {best_fitness:.6f} with {best_moves} movement operations. "
                   f"Employs balanced greedy/stochastic selection strategy with 70% greedy exploration "
                   f"and recent movement tracking for enhanced performance."
    )
    
    print(f"\nACADEMIC SUBMISSION DETAILS:")
    print(f"Submission file: {output_file}")
    print(f"Challenge identifier: {challenge_id}")
    print(f"Problem identifier: {problem_id}")
    print(f"Decision vector length: {len(decision_vector)}")
    print(f"Decision vector preview: {decision_vector[:20]}...")
    
    print(f"\n" + "=" * 80)
    print("GREEDY HEURISTIC OPTIMIZATION RESEARCH COMPLETED")
    print("Results available for comparative algorithmic studies")
    print("=" * 80)
    
    return output_file

if __name__ == "__main__":
    submission_file = main()