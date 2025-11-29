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

    def get_nix(self):
        """
        Get number of integer variables in the chromosome/decision vector.

        Returns:
            int: number of integer variables.
        """
        # the chromosome exists solely of integer variables.
        return self.setup['max_cmds'] * 2 + 1

    def fitness(self, chromosome, initial_configuration=None, verbose=False):
        """
        Fitness function for the UDP

        Args:
            chromosome: the chromosome/decision vector to be tested
            verbose: whether to provide more additional output during chromosome evaluation
        Returns:
            score: the score/fitness for this chromosome.
        """
        # By default, we start from the initial configuration provided in the problem.
        # This is also the point fitness evaluation on the optimize platform starts from.
        # During optimization, you can change the initial configuration to start,
        # e.g., from intermediate configurations.
        # Note: we do not check here whether a custom cube configuration is valid
        # (i.e., whether all cubes are connected with each other).
        if initial_configuration is None:
            initial_config_path = os.path.join(self.root_dir, self.setup['path'], 'Initial_Config.npy')
            initial_configuration = np.load(initial_config_path)

        # Create the cube ensemble with an initial cube configuration.
        cubes = ProgrammableCubes(initial_configuration)

        # Roll-out the command sequence and calculate the final fitness.
        steps_needed = cubes.apply_chromosome(chromosome, verbose)
        self.final_cube_positions = cubes.cube_position
        steps_fraction = steps_needed / self.setup['max_cmds']
        score = fitness_function(cubes, steps_fraction, self.setup['fitness_offset'],
                                 self.setup['num_cube_types'], self.initial_cube_types,
                                 self.target_cube_types, self.target_cube_positions)

        return [-score]

    def pretty(self, chromosome):
        self.fitness(chromosome, verbose=True)

    def example(self):
        pass

    def plot(self, which_one, cube_type_to_plot=[0], custom_config=None, custom_cube_types=None):
        '''
        Plot the cube ensemble with the default colour scheme given in /problems/*.yaml.
        Renders the plot without returning anything.

        Args:
            which_one: What to plot. Options: ['ensemble', 'target'].
                       'target' plots the target cube positions, i.e., the target shape.
                       'ensemble' plots the configuration after applying a chromosome (starting from the initial configuration).
            cube_type_to_plot: list of cube types to plot (int from 0 to maximum number of cube types - 1).
            custom_config: If not None, this will overwrite the cube positions used for plotting.
            custom_cube_types: If not None, this will overwrite the cube types used for plotting.
        '''
        if which_one == 'ensemble':
            assert (self.final_cube_positions is not None)
            positions = self.final_cube_positions
            cube_types = self.initial_cube_types
        if which_one == 'target':
            positions = self.target_cube_positions
            cube_types = self.target_cube_types
        if custom_config is not None:
            positions = custom_config
        if custom_cube_types is not None:
            cube_types = custom_cube_types

        cube_tensor = np.zeros((len(cube_type_to_plot),
                                self.setup['plot_dim'],
                                self.setup['plot_dim'],
                                self.setup['plot_dim']))
        offset = int(np.fabs(np.min(positions)))

        for l in range(len(cube_type_to_plot)):
            for pos in positions[cube_types == cube_type_to_plot[l]]:
                i, j, k = pos
                cube_tensor[l][i + offset][j + offset][k + offset] = 1

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_facecolor('white')
        for i in range(len(cube_type_to_plot)):
            ax.voxels(cube_tensor[i], facecolor=self.setup['colours'][cube_type_to_plot[i]], edgecolors='k', alpha=.4)

        plt.tight_layout()
        # plt.show()
