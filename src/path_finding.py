from typing import List, Tuple, Set
import math
import random
import RNA
import numpy as np

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
# Evolutionary Heuristic #
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

######################
# Indirect Heuristic #
######################

DEFAULT_PARAMS = {
            'base_random_chance': 0.1,
            'tabu_size': 15,
            'beam_width': 1,
            'max_random_chance': 0.5,
            'direct_move_threshold': 100
}

def find_indirect_path(start: RNAStructure, end: RNAStructure, fc=None, 
                      # Path search parameters
                      max_iterations=2000,
                      # Randomness control parameters  
                      random_params=DEFAULT_PARAMS) -> List[RNAStructure]:
    """
    Find an indirect path between two RNA structures using a tabu search heuristic.
    
    Args:
        start: Starting RNA structure
        end: Target RNA structure
        fc: ViennaRNA fold compound (if None, will be created)
        max_iterations: Maximum number of iterations to prevent infinite loops
        random_params: Dictionary of parameters controlling randomness and exploration:
            - base_random_chance: Base probability of making a random move (default: 0.1)
            - tabu_size: Number of recently visited structures to avoid (default: 15)
            - beam_width: Number of top moves to consider in beam search (default: 1)
            - max_random_chance: Maximum random move probability when stuck (default: 0.5)
            - direct_move_threshold: Iterations without progress before attempting direct moves (default: 100)
            
    Returns:
        List of structures forming the path from start to end
    """
    if start.sequence != end.sequence:
        raise ValueError("Structures must have the same sequence")
    
    # Create fold compound if not provided
    if fc is None:
        fc = RNA.fold_compound(start.sequence)
    
    # Set default randomness parameters if not provided
    if random_params is None:
        random_params = {}
    
    # Randomness and exploration parameters with defaults
    base_random_chance = random_params.get('base_random_chance')
    tabu_size = random_params.get('tabu_size')
    beam_width = random_params.get('beam_width')
    max_random_chance = random_params.get('max_random_chance')
    direct_move_threshold = random_params.get('direct_move_threshold')
    
    # Current randomness level (may change during search)
    current_random_chance = base_random_chance
    
    # Initialize path with start structure
    path = [start]
    current = RNAStructure(start.sequence, base_pairs=start.base_pairs.copy())
    current_db = current.to_dotbracket()
    end_db = end.to_dotbracket()
    
    # Calculate base pair distance between start and end
    base_pair_diff = len(start.base_pairs.symmetric_difference(end.base_pairs))
    
    # Dynamically adjust max_iterations based on the complexity
    max_iterations = max(max_iterations, base_pair_diff * 5)
    
    # Memory structures to avoid cycles
    tabu_list = set([current_db])
    visited_structures = {current_db: 0}  # structure -> position in path
    
    # Progress tracking
    iterations = 0
    no_progress_count = 0
    initial_distance = structure_distance(current_db, end_db)
    best_distance_so_far = initial_distance
    
    # Cycle detection
    cycle_detection_window = min(50, max_iterations // 10)
    recent_distances = []
    
    # Main search loop
    while current.base_pairs != end.base_pairs and iterations < max_iterations:
        iterations += 1
        
        # ----- STEP 1: Generate and filter possible moves -----
        
        # Generate all possible moves from current structure
        moves = generate_moves(current.sequence, current_db)
        if not moves:
            break  # No moves possible
            
        # Filter out moves that would lead to structures in the tabu list
        valid_moves = [(new_struct, move_type, pos) for new_struct, move_type, pos in moves 
                      if new_struct not in tabu_list]
        
        # If all moves lead to tabu structures, allow revisiting older structures
        if not valid_moves and moves:
            # Consider structures that were visited at least 20 steps ago or not at all
            valid_moves = [(new_struct, move_type, pos) for new_struct, move_type, pos in moves 
                          if new_struct not in visited_structures or 
                          (iterations - visited_structures[new_struct]) > 20]
            
            # If still no valid moves, use all moves
            if not valid_moves:
                valid_moves = moves
        
        if not valid_moves:
            break
            
        # ----- STEP 2: Calculate current state metrics -----
        
        current_energy = fc.eval_structure(current_db)
        current_distance = structure_distance(current_db, end_db)
        
        # Update best distance seen so far
        if current_distance < best_distance_so_far:
            best_distance_so_far = current_distance
        
        # ----- STEP 3: Detect and handle cycling behavior -----
        
        # Track recent distances for cycle detection
        recent_distances.append(current_distance)
        if len(recent_distances) > cycle_detection_window:
            recent_distances.pop(0)
        
        # If variance is very low after enough iterations, we might be cycling
        if len(recent_distances) == cycle_detection_window:
            distance_variance = np.var(recent_distances)
            if distance_variance < 1.0 and iterations > 100:
                # Make multiple random moves to break out of the cycle
                for _ in range(min(5, len(valid_moves))):
                    if valid_moves:
                        random_move = random.choice(valid_moves)
                        new_struct, move_type, (i, j) = random_move
                        
                        # Apply the move
                        if move_type == 'add':
                            current.add_pair(i-1, j-1)
                        else:
                            current.remove_pair(i-1, j-1)
                            
                        current_db = new_struct
                        path.append(RNAStructure(current.sequence, base_pairs=current.base_pairs.copy()))
                        
                        # Update memory structures
                        tabu_list.add(current_db)
                        if len(tabu_list) > tabu_size:
                            tabu_list.remove(next(iter(tabu_list)))
                        visited_structures[current_db] = iterations
                
                # Reset cycle detection
                recent_distances = []
                continue
        
        # ----- STEP 4: Decide whether to make a random move -----
        
        # Adjust random chance based on progress
        if no_progress_count > 10:
            # Increase randomness when stuck
            current_random_chance = min(max_random_chance, 
                                       base_random_chance * (1 + (no_progress_count / 10)))
        else:
            # Gradually return to base level
            current_random_chance = max(base_random_chance, current_random_chance * 0.95)
        
        # Decide whether to make a random move
        make_random_move = random.random() < current_random_chance
        
        # ----- STEP 5: Select the next move -----
        
        if make_random_move:
            # Random selection
            selected_move = random.choice(valid_moves)
        else:
            # Beam search selection
            candidates = []
            
            for new_struct, move_type, (i, j) in valid_moves:
                # Calculate metrics for the new structure
                new_energy = fc.eval_structure(new_struct)
                new_distance = structure_distance(new_struct, end_db)
                
                # Convert move to base pair change for direct distance calculation
                if move_type == 'add':
                    new_bp = current.base_pairs.copy()
                    new_bp.add((i-1, j-1))
                else:
                    new_bp = current.base_pairs.copy()
                    new_bp.remove((i-1, j-1))
                
                # Calculate base pair distance directly
                bp_distance = len(end.base_pairs.symmetric_difference(new_bp))
                
                # Calculate key factors for scoring
                energy_change = new_energy - current_energy
                distance_improvement = current_distance - new_distance
                bp_distance_improvement = len(end.base_pairs.symmetric_difference(current.base_pairs)) - bp_distance
                
                # Combined distance factor (weighted toward base pair distance)
                distance_factor = bp_distance_improvement * 2 + distance_improvement
                
                # Balance energy vs. distance - penalize energy increases more if they don't improve distance
                energy_weight = 3.0 if distance_factor <= 0 else 1.0
                
                # Final score (lower is better)
                score = energy_change * energy_weight - distance_factor
                
                # Penalize revisiting structures
                if new_struct in visited_structures:
                    steps_since_visit = iterations - visited_structures[new_struct]
                    revisit_penalty = 10.0 / (1.0 + steps_since_visit/10.0)
                    score += revisit_penalty
                
                candidates.append((score, (new_struct, move_type, (i, j))))
            
            # Sort candidates by score (lower is better)
            candidates.sort(key=lambda x: x[0])
            
            # Select probabilistically from top candidates
            if candidates:
                # Take top beam_width candidates
                top_candidates = candidates[:beam_width]
                
                # Convert scores to probabilities (better scores get higher probability)
                total_score = sum(1.0 / (1.0 + abs(score)) for score, _ in top_candidates)
                probs = [(1.0 / (1.0 + abs(score))) / total_score for score, _ in top_candidates]
                
                # Select based on probabilities
                r = random.random()
                cumulative = 0
                selected_idx = 0
                for i, prob in enumerate(probs):
                    cumulative += prob
                    if r <= cumulative:
                        selected_idx = i
                        break
                
                selected_move = top_candidates[selected_idx][1]
            else:
                # Fallback to random move
                selected_move = random.choice(valid_moves)
        
        # ----- STEP 6: Apply the selected move -----
        
        new_struct, move_type, (i, j) = selected_move
        
        # Update current structure
        if move_type == 'add':
            current.add_pair(i-1, j-1)  # Convert from 1-indexed to 0-indexed
        else:  # move_type == 'remove'
            current.remove_pair(i-1, j-1)
            
        current_db = new_struct
        path.append(RNAStructure(current.sequence, base_pairs=current.base_pairs.copy()))
        
        # ----- STEP 7: Update memory structures -----
        
        # Update tabu list
        tabu_list.add(current_db)
        if len(tabu_list) > tabu_size:
            tabu_list.remove(next(iter(tabu_list)))
        
        # Update visited structures
        visited_structures[current_db] = iterations
        
        # ----- STEP 8: Track progress and adjust strategy -----
        
        new_distance = structure_distance(current_db, end_db)
        if new_distance >= current_distance:
            no_progress_count += 1
        else:
            # Reset counter when making progress
            no_progress_count = 0
        
        # If stuck for too long, try more aggressive strategies
        if no_progress_count > direct_move_threshold:
            # Print progress updates periodically
            if iterations % 100 == 0:
                print(f"Iteration {iterations}, current distance: {new_distance}, best: {best_distance_so_far}")
            
            # Try to make a direct move toward the target
            current_bp = current.base_pairs
            end_bp = end.base_pairs
            
            # Find pairs to add or remove
            to_add = end_bp - current_bp
            to_remove = current_bp - end_bp
            
            direct_move_made = False
            
            # First try to remove a pair that's not in the target
            for i, j in to_remove:
                for new_struct, move_type, (m, n) in valid_moves:
                    if move_type == 'remove' and m-1 == i and n-1 == j:
                        # Apply this move
                        current.remove_pair(i, j)
                        current_db = new_struct
                        path.append(RNAStructure(current.sequence, base_pairs=current.base_pairs.copy()))
                        tabu_list.add(current_db)
                        visited_structures[current_db] = iterations
                        direct_move_made = True
                        break
                if direct_move_made:
                    break
            
            # If no removal was possible, try to add a pair from the target
            if not direct_move_made:
                for i, j in to_add:
                    for new_struct, move_type, (m, n) in valid_moves:
                        if move_type == 'add' and m-1 == i and n-1 == j:
                            # Apply this move
                            current.add_pair(i, j)
                            current_db = new_struct
                            path.append(RNAStructure(current.sequence, base_pairs=current.base_pairs.copy()))
                            tabu_list.add(current_db)
                            visited_structures[current_db] = iterations
                            direct_move_made = True
                            break
                    if direct_move_made:
                        break
            
            # Reset progress counter if we made a direct move
            if direct_move_made:
                no_progress_count = 0
            else:
                # If still stuck, increase randomness significantly
                current_random_chance = min(0.9, current_random_chance * 1.5)
                no_progress_count = 0
            
        # ----- STEP 9: Check if we've reached the target -----
        if current.base_pairs == end.base_pairs:
            break
            
    # ----- Finalize path if we didn't reach the target -----
    if current.base_pairs != end.base_pairs and iterations >= max_iterations:
        print(f"Warning: Indirect path search reached iteration limit ({max_iterations}). "
              f"Final distance: {structure_distance(current_db, end_db)}")
        
        # If we're halfway there, try a direct path to finish
        if structure_distance(current_db, end_db) < initial_distance / 2:
            direct_finish = find_direct_path(current, end)
            if direct_finish and len(direct_finish) > 1:
                path.extend(direct_finish[1:])  # Skip first element (duplicate)
    
    return path

def find_best_indirect_path(start: RNAStructure, end: RNAStructure, fc=None, num_attempts=5) -> Tuple[List[RNAStructure], float]:
    """
    Find the best indirect path by running the indirect path finder multiple times
    with different parameters and selecting the path with the lowest energy barrier.
    
    Args:
        start: Starting RNA structure
        end: Target RNA structure
        fc: ViennaRNA fold compound (if None, will be created)
        num_attempts: Number of attempts to run the indirect path finder
        
    Returns:
        Tuple containing (best path, energy barrier)
    """
    # Create fold compound if not provided
    if fc is None:
        try:
            fc = RNA.fold_compound(start.sequence)
        except Exception as e:
            print(f"Error creating fold compound: {e}")
            raise
    
    # Calculate base pair distance to assess complexity
    bp_distance = len(start.base_pairs.symmetric_difference(end.base_pairs))
    print(f"Base pair distance between structures: {bp_distance}")
    
    # Adjust number of attempts based on complexity
    adjusted_attempts = min(num_attempts, 3 + int(bp_distance / 10))
    
    best_path = None
    lowest_barrier = float('inf')
    
    # Calculate start energy once
    start_energy = fc.eval_structure(start.to_dotbracket())
    
    # Predefined parameter sets for different search strategies
    parameter_sets = [
        # Balanced approach
        DEFAULT_PARAMS,
        # More exploration
        {
            'base_random_chance': 0.4,
            'tabu_size': 20,
            'beam_width': 2,
            'max_random_chance': 0.9,
            'direct_move_threshold': 80
        },
        # More exploitation
        {
            'base_random_chance': 0.1,
            'tabu_size': 10,
            'beam_width': 5,
            'max_random_chance': 0.6,
            'direct_move_threshold': 120
        }
    ]
    
    # Try different parameter combinations
    for attempt in range(adjusted_attempts):
        print(f"Starting indirect path attempt {attempt+1}/{adjusted_attempts}")
        
        # Use predefined parameter sets or generate random ones
        if attempt < len(parameter_sets):
            random_params = parameter_sets[attempt]
        else:
            # Generate random parameters for diversity
            random_params = {
                'base_random_chance': 0.1 + (0.3 * random.random()),
                'tabu_size': random.randint(10, 25),
                'beam_width': random.randint(2, 5),
                'max_random_chance': 0.7 + (0.2 * random.random()),
                'direct_move_threshold': random.randint(80, 120)
            }
        
        # For complex structures, increase iterations
        max_iterations = 1000 + bp_distance * 10
        
        try:
            path = find_indirect_path(
                start, end, fc, 
                max_iterations=max_iterations,
                random_params=random_params
            )
            
            # Check if path reaches the target
            if path[-1].base_pairs != end.base_pairs:
                print(f"Warning: Attempt {attempt+1} failed to reach target structure")
                continue
            
            # Calculate energy barrier
            max_energy = start_energy
            
            for structure in path:
                energy = fc.eval_structure(structure.to_dotbracket())
                max_energy = max(max_energy, energy)
            
            barrier = max_energy - start_energy
            
            print(f"Attempt {attempt+1}: path length = {len(path)}, energy barrier = {barrier:.2f}")
            
            if barrier < lowest_barrier:
                lowest_barrier = barrier
                best_path = path
                print(f"New best barrier: {barrier:.2f}")
        except Exception as e:
            print(f"Error in attempt {attempt+1}: {e}")
    
    if best_path is None:
        print("Warning: All indirect path attempts failed. Falling back to direct path.")
        best_path = find_direct_path(start, end)
        
        # Calculate barrier for direct path
        max_energy = start_energy
        for structure in best_path:
            energy = fc.eval_structure(structure.to_dotbracket())
            max_energy = max(max_energy, energy)
        lowest_barrier = max_energy - start_energy
        print(f"Direct path barrier: {lowest_barrier:.2f}")
    
    return best_path
