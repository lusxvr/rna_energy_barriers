from typing import List, Tuple, Set
import math
import random

from src.rna_structure import RNAStructure
import src.utils as utils

# Global constants (R in kcal/(mol*K), T in Kelvin)
R_CONST = 0.0019872   # kcal/(mol*K)
T_CONST = 410.15      # ~37°C

def find_direct_path(start: RNAStructure, end: RNAStructure) -> List[RNAStructure]:
    """
    Find a direct path between two RNA structures by first removing pairs from start
    that aren't in end, then adding pairs from end that aren't in start.
    
    Args:
        start: Starting RNA structure
        end: Target RNA structure
        
    Returns:
        List of structures forming the path from start to end
    """
    if start.sequence != end.sequence:
        raise ValueError("Structures must have the same sequence")
        
    path = [start]
    current = RNAStructure(start.sequence, base_pairs=start.base_pairs.copy())
    
    # First remove pairs that are in start but not in end
    pairs_to_remove = start.base_pairs - end.base_pairs
    for i, j in pairs_to_remove:
        current.remove_pair(i, j)
        path.append(RNAStructure(current.sequence, base_pairs=current.base_pairs.copy()))
        
    # Then add pairs that are in end but not in start
    pairs_to_add = end.base_pairs - start.base_pairs
    for i, j in pairs_to_add:
        current.add_pair(i, j)
        path.append(RNAStructure(current.sequence, base_pairs=current.base_pairs.copy()))
        
    return path

######################
# Indirect Heuristic #
######################

def generate_moves(seq, structure):
    """
    Given a sequence and a current structure (in dot-bracket notation),
    generate every valid outcome with one base move (either addition or removal of a single base pair).
    
    Returns a list of tuples: (new_structure, move_type, (i, j))
    where move_type is either 'add' or 'remove', and (i, j) indicates the base pair changed.
    Positions are 1-indexed.
    """
    moves = []
    n = len(structure)
    # Convert structure to a pair table (1-indexed)
    pt = utils.dotbracket_to_pt(structure)
    
    # --- Removal moves: remove an existing base pair ---
    # For every base i that is paired (and to avoid duplicates, ensure i < pt[i])
    for i in range(1, n+1):
        j = pt[i]
        if j > i:
            # Create new structure by removing the pair (i, j)
            new_struct = list(structure)
            new_struct[i-1] = '.'
            new_struct[j-1] = '.'
            moves.append((''.join(new_struct), 'remove', (i, j)))
    
    # --- Addition moves: add one new base pair ---
    # We consider all candidate pairs (i, j) with j - i >= 4 (minimum hairpin length constraint)
    for i in range(1, n+1):
        # Only consider i if currently unpaired
        if pt[i] != 0:
            continue
        for j in range(i+3, n+1):
            if pt[j] != 0:
                continue
            # Check if the bases are complementary
            if not utils.complementary(seq[i-1], seq[j-1]):
                continue
            
            # Check that adding (i, j) does not cross any existing base pair.
            # For each existing base pair (k, l) (with k < l), ensure that both k and l lie entirely inside [i, j]
            # or entirely outside [i, j]. If one lies inside and the other outside, the new pair would cross.
            conflict = False
            for k in range(1, n+1):
                l = pt[k]
                if l > k:  # (k, l) is an existing pair
                    in_i = (i < k < j)
                    in_j = (i < l < j)
                    if in_i != in_j:
                        conflict = True
                        break
            if conflict:
                continue
            
            # Otherwise, adding (i, j) yields a valid structure.
            new_struct = list(structure)
            new_struct[i-1] = '('
            new_struct[j-1] = ')'
            moves.append((''.join(new_struct), 'add', (i, j)))
    
    return moves

def structure_distance(s1, s2):
    """
    Compute a simple Hamming distance between two dot-bracket strings.
    Assumes both strings are of the same length.
    """
    if len(s1) != len(s2):
        raise ValueError("Structures must be of equal length")
    return sum(1 for a, b in zip(s1, s2) if a != b)

def move_probabilities(moves, end_struct, fc, T=T_CONST, beta=0):
    """
    For each move in 'moves' (list of tuples (new_structure, move_type, (i, j))),
    calculates the free energy using ViennaRNA's fold compound 'fc' and returns
    the probability associated with each move according to the Boltzmann law.
    
    Returns a list of probabilities corresponding to the moves.
    """
    energies = []
    for new_struct, move_type, pos in moves:
        # Calculate the free energy of the resulting structure (in kcal/mol)
        E = fc.eval_structure(new_struct)
        energies.append(E)

    for k,(new_struct, move_type, pos) in enumerate(moves):
        m = structure_distance(new_struct, end_struct)
        energies[k] += beta*m
    
    # Determine the minimum energy to factor the exponentials
    E_min = min(energies)
    
    # Calculate Boltzmann weights by factoring out the minimum energy.
    # Thus, weight = exp(- (E - E_min) / (R_CONST * T_CONST))
    weights = [math.exp(-(E - E_min) / (R_CONST * T)) for E in energies]
    
    total = sum(weights)
    probabilities = [w / total for w in weights]
    return probabilities

def select_next_structure(seq, current_structure, end_struct, fc, T=T_CONST, beta=0):
    """
    Given a sequence and current structure, generate all one-base moves,
    compute their Boltzmann probabilities, and randomly select one move according
    to these probabilities.
    
    Returns the new structure and the move information (move_type, positions).
    If no moves are available, returns the current structure and None.
    """
    moves = generate_moves(seq, current_structure)
    if not moves:
        return current_structure, None
    probs = move_probabilities(moves, end_struct, fc, T=T, beta=beta)
    r = random.random()
    cumulative = 0.0
    for move, prob in zip(moves, probs):
        cumulative += prob
        if r <= cumulative:
            return move[0], (move[1], move[2])
    return moves[-1][0], (moves[-1][1], moves[-1][2])