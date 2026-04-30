"""
Genetic Algorithm for Feature Selection
"""
import numpy as np


def initialize_population(population_size: int, n_features: int) -> np.ndarray:
    """
    Creates an initial population of random binary chromosomes
    (each chromosome has at least one '1').
    """
    population = np.random.randint(0, 2, size=(population_size, n_features))

    for i in range(population_size):
        if population[i].sum() == 0:
            population[i][np.random.randint(0, n_features)] = 1

    return population



def tournament_selection(
    population: np.ndarray,
    fitness_scores: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """
    Randomly selects k individuals, returns the best one
    """
    pop_size = len(population)
    candidates_idx = np.random.choice(pop_size, size=k, replace=False)
    best_idx = candidates_idx[np.argmax(fitness_scores[candidates_idx])]
    return population[best_idx].copy()

def crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_rate: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single-point crossover between two parents with a given
    probability to produce two children.
    """
    if np.random.rand() > crossover_rate:
        return parent1.copy(), parent2.copy()

    n = len(parent1)
    point = np.random.randint(1, n)

    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])

    return child1, child2

def mutation(chromosome: np.ndarray, mutation_rate: float = 0.02) -> np.ndarray:
    """
    Inverts each bit with a given probability
    while ensuring at least one feature remains selected.
    """
    mutated = chromosome.copy()
    flip_mask = np.random.rand(len(mutated)) < mutation_rate
    mutated[flip_mask] ^= 1  # XOR: 0→1, 1→0

    if mutated.sum() == 0:
        mutated[np.random.randint(0, len(mutated))] = 1

    return mutated
