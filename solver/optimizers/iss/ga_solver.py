#!/usr/bin/env python3
"""
Enhanced Genetic Algorithm for ISS Spacecraft Assembly Problem
Programmable Cubes Challenge - GECCO 2024 Space Optimisation Competition (SpOC)

This module implements an advanced genetic algorithm with comprehensive experimental
documentation, visualization capabilities, and result analysis for the International
Space Station spacecraft assembly optimization problem. The algorithm features
adaptive mechanisms, intelligent initialization strategies, and systematic
performance monitoring for competitive optimization results.

The genetic algorithm employs multi-strategy population initialization, tournament
selection, adaptive crossover and mutation operators, elite preservation, and
local search enhancement. Fitness direction optimization ensures proper convergence
toward negative fitness values, indicating superior assembly configurations.

Key algorithmic enhancements include:
- Corrected fitness direction optimization (negative values indicate better solutions)
- Inverse-move cleanup for chromosome efficiency optimization
- Adaptive mutation rate mechanisms responding to optimization stagnation
- Comprehensive experimental data collection and academic visualization

Target Performance: Achieve fitness of -0.991 or superior (championship-level performance)

Usage:
    python solver/optimizers/iss/ga_solver.py

Dependencies:
    - numpy: Numerical computing and array operations
    - matplotlib: Result visualization and plotting
    - tqdm: Progress monitoring during optimization
    - scipy: Distance calculations and scientific computing
    - json: Experimental data serialization
"""

import sys
import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for automated plot generation
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime
from tqdm import tqdm
from scipy.spatial.distance import cdist
from collections import defaultdict

# Add the src directory and the repository root to the Python path
repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))
