#!/usr/bin/env python3

import sys
import os
import numpy as np
import random
import time
from tqdm import tqdm

# Configure Python path to load the project modules
repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))

from src.programmable_cubes_UDP import programmable_cubes_UDP

# Parameters
N_ITERATIONS = 1000
RANDOM_SEED = 42
MAX_CHROMOSOME_LENGTH = 200


def generate_random_chromosome(num_cubes, max_length=100):
    length = random.randint(1, max_length)
    chromosome = []

    for _ in range(length):
        cube_id = random.randint(0, num_cubes - 1)
        move_command = random.randint(0, 5)
        chromosome.extend([cube_id, move_command])

    chromosome.append(-1)
    return np.array(chromosome)


def evaluate_chromosome(udp, chromosome):
    try:
        fitness = udp.fitness(chromosome)
        return fitness[0]
    except:
        return float('-inf')


def count_moves(chromosome):
    end_pos = np.where(chromosome == -1)[0][0]
    return end_pos // 2


def random_search_iss():
    print("=== RANDOM SEARCH FOR ISS ASSEMBLY (MINIMAL VERSION) ===")

    # Seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load Problem
    udp = programmable_cubes_UDP('ISS', repo_root)

    num_cubes = udp.setup['num_cubes']
    max_cmds = udp.setup['max_cmds']

    print(f"Number of cubes: {num_cubes}")
    print(f"Max commands: {max_cmds}")
    print(f"Iterations: {N_ITERATIONS}")
    print()

    best_fitness = float('inf')
    best_chromosome = None
    best_moves = 0

    start_time = time.time()

    for _ in tqdm(range(N_ITERATIONS), desc="Random Search"):
        chrom = generate_random_chromosome(num_cubes, max_length=min(200, max_cmds))
        fitness = evaluate_chromosome(udp, chrom)

        if fitness < best_fitness:
            best_fitness = fitness
            best_chromosome = chrom
            best_moves = count_moves(chrom)

    runtime = time.time() - start_time

    print("\n=== RESULTS ===")
    print(f"Best fitness: {best_fitness}")
    print(f"Moves: {best_moves}")
    print(f"Chromosome length: {len(best_chromosome)}")
    print(f"Runtime: {runtime:.2f} sec")
    print(f"Iterations/sec: {N_ITERATIONS / runtime:.1f}")

    print("Best Chromosome:")
    for i in range(len(best_chromosome)):
        print(best_chromosome[i], end=" ")
        if (i + 1) % 10 == 0:
            print()
    print()

    print("\n=== DONE ===")
    return best_chromosome, best_fitness, best_moves


if __name__ == "__main__":
    random_search_iss()
