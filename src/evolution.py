import random
import RNA

import src.path_finding as pf

def init_population(seq, start_struct, N):
    """
    Initiate a population of size N.
    Each individual is a dictionary with keys:
      'structure': the current dot-bracket structure
      'distance': the current distance to the end structure (initialized to None)
    Since sequence remains fixed, we only evolve the structure.
    """
    population = []
    for _ in range(N):
        individual = {
            'structure': start_struct,
            'distance': 1,
            'highest_energy': 0,
            'path': [start_struct],
            'energy_profile': [0]  # Will be set to actual energy later
        }
        population.append(individual)
    return population

def evolve_population_inclusive(population, seq, fc, start_struct, end_struct, alpha=0.7, T=pf.T_CONST, beta=0):
    """
    For each individual in the current population, perform one evolutionary step
    to generate a new candidate via a one-base move. Then, combine the new candidates
    with the original population and select the best N individuals (those with the smallest
    distance to the end structure).
    
    Returns the new population.
    """
    N = len(population)
    new_candidates = []
    
    # Generate new candidates for each individual
    for individual in population:
        current_struct = individual['structure']
        next_struct, move_info = pf.select_next_structure(seq, current_struct, end_struct, fc, T=T, beta=beta)
        dist = pf.structure_distance(next_struct, end_struct)
        
        # Track the path taken
        path = individual.get('path', [start_struct])
        energy_profile = individual.get('energy_profile', [fc.eval_structure(start_struct)])
        
        candidate = {
            'structure': next_struct,
            'distance': dist,
            'highest_energy': max(individual['highest_energy'], fc.eval_structure(next_struct)-fc.eval_structure(start_struct)),
            'path': path + [next_struct],
            'energy_profile': energy_profile + [fc.eval_structure(next_struct)]
        }
        new_candidates.append(candidate)
    
    # Update original population with their current distance
    current_with_distance = []
    for ind in population:
        d = pf.structure_distance(ind['structure'], end_struct)
        current_with_distance.append({
            'structure': ind['structure'],
            'distance': d,
            'highest_energy': ind['highest_energy'],
            'path': ind.get('path', [start_struct]),
            'energy_profile': ind.get('energy_profile', [fc.eval_structure(start_struct)])
        })
    
    # Combine the old population with the new candidates
    combined = new_candidates + current_with_distance
    
    # Sort by distance (lower is better) and keep the best N individuals
    combined.sort(key=lambda ind: (ind['distance'], ind['highest_energy']))
    #new_population = combined[:N]

    best_count = int(alpha * N)
    random_count = N - best_count

    # Keep the best alpha% individuals
    best_individuals = combined[:best_count]

    # Select N/3 individuals at random among the remaining ones
    remaining_individuals = combined[N:]
    random_individuals = random.sample(remaining_individuals, random_count) if remaining_individuals else []

    # Create the new population by combining the best and random individuals
    new_population = best_individuals + random_individuals
    
    return new_population

def compute_barrier(population):
    """
    Compute the highest energy barrier encountered by the population.
    """
    for ind in population:
        if ind['distance'] == 0:
            return ind['highest_energy']
        
def best_folding(seq, start_struct, end_struct, alpha = 0.7, N = 100, max_steps = 100, beta=0, T=pf.T_CONST):
    """
    Given a sequence, start and end structures, evolve a population of structures
    to find the one with the smallest distance to the end structure.
    """
    # Create a fold compound for the sequence (ViennaRNA initialization)
    fc = RNA.fold_compound(seq)

    # 1. Initialize population of size N
    population = init_population(seq, start_struct, N)

    step = 0
    while step<max_steps and compute_barrier(population) is None:
        step += 1
        population = evolve_population_inclusive(population, seq, fc, start_struct, end_struct, alpha=alpha, T=T, beta=beta)
        if step % 10 == 0:
            print(f"\nGeneration {step}:")
            print(population[0])
    
    best = min(population, key=lambda ind: (ind['distance'], ind['highest_energy']))
    return best, step