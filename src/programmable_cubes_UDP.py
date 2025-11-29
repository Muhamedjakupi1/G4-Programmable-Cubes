# Programmable Cubes challenge
# GECCO 2024 Space Optimisation Competition (SpOC)

import os
from numba import njit
import numpy as np
from numba.typed import List
# from src.CubeMoveset import MoveSetRev // kur te bon ledi fajllin e perdorim
import json
import matplotlib.pyplot as plt


############################################################################################################################################
##### THE UDP DEFINING AND IMPLEMENTING THE OPTIMIZATION CHALLENGE
############################################################################################################################################

class programmable_cubes_UDP:
    def __init__(self, problem, root_dir='.'):
        """
        A Pygmo compatible UDP User Defined Problem representing the Programmable Cubes challenge for SpOC 2024.

        Args:
            problem: Name of the problem / scenario to be used. Implemented: ISS, JWST, Enterprise.
            root_dir: Root directory of the project (default: current directory)
        """
        # Variable name used for storing cube ensemble configuration after chromosome evaluation
        self.final_cube_positions = None

        # Store root directory
        self.root_dir = root_dir

        # Load specifications of the problem
        problem_path = os.path.join(root_dir, 'problems', f'{problem}.json')
        with open(problem_path, 'r') as infile:
            self.setup = json.load(infile)

        # Load target cube locations and cube types
        data_path = os.path.join(root_dir, self.setup['path'])
        self.target_cube_positions = np.load(os.path.join(data_path, 'Target_Config.npy'))
        self.target_cube_types = np.load(os.path.join(data_path, 'Target_Cube_Types.npy'))
        # Load cube types of initial configuration
        self.initial_cube_types = np.load(os.path.join(data_path, 'Initial_Cube_Types.npy'))

    def get_bounds(self):
        """
        tldr; Get bounds for the decision variables.

        The chromosome is composed of tuples representing cube ID and pivot command,

        i.e.,

        chromosome = [cube ID, command ID, cube ID, command ID, ..., -1]

        The chromosome always has to end with -1. It is read in from left to right, executing for each cube ID the
        command ID that comes right afterwards.
        An early termination while reading the chromosome is possible by setting -1 instead of a valid cube ID.

        Returns:
            Tuple of lists: bounds for the decision variables.
        """
        # lb = lower bounds of the chromosome entries
        # rb = upper bounds of the chromosome entries
        lb, rb = [], []

        # All cube IDs are bound below by -1 (the first entry of -1 indicates the chromosome end, i.e.,
        # everything that comes after is ignored when calculating the fitness).
        # Command IDs are bound below by 0.
        # The chromosome has to end with at least one entry of -1 (added here to ensure always a valid chromosome).
        lb += [-1, 0] * self.setup['max_cmds']
        lb += [-1]

        # Maximum cube ID is the number of cubes - 1.
        # Moves go from 0 to 5 (clock - and counterclockwise pivoting around x,y,z axis).
        rb += [self.setup['num_cubes'] - 1, 5] * self.setup['max_cmds']
        rb += [-1]

        return (lb, rb)