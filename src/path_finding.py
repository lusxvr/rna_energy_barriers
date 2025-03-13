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

def find_indirect_path(start: RNAStructure, end: RNAStructure, fc=None, max_iterations=2000, tabu_size=15, random_chance=0.1) -> List[RNAStructure]:
    """
    Find an indirect path between two RNA structures using a tabu search heuristic.
    Keeps track of recently visited structures to avoid cycles and occasionally makes
    random moves to escape local minima.
    
    Args:
        start: Starting RNA structure
        end: Target RNA structure
        fc: ViennaRNA fold compound (if None, will be created)
        max_iterations: Maximum number of iterations to prevent infinite loops
        tabu_size: Number of recently visited structures to avoid
        random_chance: Probability of making a random move instead of the best move
        
    Returns:
        List of structures forming the path from start to end
    """
    if start.sequence != end.sequence:
        raise ValueError("Structures must have the same sequence")
    
    # Create fold compound if not provided
    if fc is None:
        fc = RNA.fold_compound(start.sequence)
    
    path = [start]
    current = RNAStructure(start.sequence, base_pairs=start.base_pairs.copy())
    current_db = current.to_dotbracket()
    end_db = end.to_dotbracket()
    
    # Tabu list to avoid revisiting recent structures (using dot-bracket notation as keys)
    tabu_list = set([current_db])
    # Track all visited structures to detect longer cycles
    visited_structures = {current_db: 0}  # structure -> position in path
    
    # Calculate base pair distance between start and end
    base_pair_diff = len(start.base_pairs.symmetric_difference(end.base_pairs))
    
    # Dynamically adjust max_iterations based on the base pair difference
    # For complex RNA with more differences, we need more iterations
    max_iterations = max(max_iterations, base_pair_diff * 5)
    
    iterations = 0
    no_progress_count = 0
    initial_distance = structure_distance(current_db, end_db)
    best_distance_so_far = initial_distance
    
    # For tracking if we're going in circles
    cycle_detection_window = min(50, max_iterations // 10)
    recent_distances = []
    
    while current.base_pairs != end.base_pairs and iterations < max_iterations:
        iterations += 1
        
        # Generate all possible moves from current structure
        moves = generate_moves(current.sequence, current_db)
        if not moves:
            break  # No moves possible
            
        # Filter out moves that would lead to structures in the tabu list
        valid_moves = [(new_struct, move_type, pos) for new_struct, move_type, pos in moves 
                      if new_struct not in tabu_list]
        
        # If all moves lead to tabu structures, consider the least recently visited ones
        if not valid_moves and moves:
            # Allow moves to structures visited a long time ago
            oldest_visited = float('inf')
            for new_struct, move_type, pos in moves:
                if new_struct in visited_structures:
                    visit_time = visited_structures[new_struct]
                    if visit_time < oldest_visited:
                        oldest_visited = visit_time
            
            # Consider structures that were visited at least 20 steps ago or not at all
            valid_moves = [(new_struct, move_type, pos) for new_struct, move_type, pos in moves 
                          if new_struct not in visited_structures or 
                          (iterations - visited_structures[new_struct]) > 20]
            
            # If still no valid moves, use all moves
            if not valid_moves:
                valid_moves = moves
        
        if not valid_moves:
            break
            
        # Calculate current energy and distance
        current_energy = fc.eval_structure(current_db)
        current_distance = structure_distance(current_db, end_db)
        
        # Update best distance
        if current_distance < best_distance_so_far:
            best_distance_so_far = current_distance
        
        # Track recent distances for cycle detection
        recent_distances.append(current_distance)
        if len(recent_distances) > cycle_detection_window:
            recent_distances.pop(0)
        
        # Check for cycling behavior - if we're revisiting similar distances repeatedly
        if len(recent_distances) == cycle_detection_window:
            distance_variance = np.var(recent_distances) if 'np' in globals() else sum((d - sum(recent_distances)/len(recent_distances))**2 for d in recent_distances) / len(recent_distances)
            # If variance is very low, we might be cycling
            if distance_variance < 1.0 and iterations > 100:
                # Force a major perturbation - make multiple random moves
                for _ in range(min(5, len(valid_moves))):
                    if valid_moves:
                        random_move = random.choice(valid_moves)
                        new_struct, move_type, (i, j) = random_move
                        
                        # Update current structure
                        if move_type == 'add':
                            current.add_pair(i-1, j-1)
                        else:
                            current.remove_pair(i-1, j-1)
                            
                        current_db = new_struct
                        path.append(RNAStructure(current.sequence, base_pairs=current.base_pairs.copy()))
                        
                        # Clear the recent distances
                        recent_distances = []
                        
                        # Update tabu list and visited structures
                        tabu_list.add(current_db)
                        if len(tabu_list) > tabu_size:
                            tabu_list.remove(next(iter(tabu_list)))
                        visited_structures[current_db] = iterations
                
                # Skip to the next iteration
                continue
        
        # Decide whether to make a random move to escape local minima
        # Increase random chance if we're stuck
        effective_random_chance = random_chance
        if no_progress_count > 10:
            effective_random_chance = min(0.8, random_chance * (1 + (no_progress_count / 10)))
        
        make_random_move = random.random() < effective_random_chance
        
        if make_random_move:
            # Select a random move
            selected_move = random.choice(valid_moves)
        else:
            # Find the best move using beam search - maintain top 3 candidates
            beam_width = 3
            candidates = []
            
            for new_struct, move_type, (i, j) in valid_moves:
                # Calculate energy of the new structure
                new_energy = fc.eval_structure(new_struct)
                
                # Calculate distance to target using both methods
                new_dot_distance = structure_distance(new_struct, end_db)
                
                # Convert move to base pair change
                if move_type == 'add':
                    new_bp = current.base_pairs.copy()
                    new_bp.add((i-1, j-1))
                else:
                    new_bp = current.base_pairs.copy()
                    new_bp.remove((i-1, j-1))
                
                # Calculate base pair distance directly
                bp_distance = len(end.base_pairs.symmetric_difference(new_bp))
                
                # Score using a weighted combination of metrics
                energy_factor = new_energy - current_energy
                dot_distance_change = current_distance - new_dot_distance
                bp_distance_change = len(end.base_pairs.symmetric_difference(current.base_pairs)) - bp_distance
                
                # Weight the changes - prioritize base pair distance over dot-bracket distance
                # And balance energy with distance
                distance_factor = bp_distance_change * 2 + dot_distance_change
                
                # Penalize energy increases more if they don't improve distance
                if distance_factor <= 0:
                    energy_weight = 3.0
                else:
                    energy_weight = 1.0
                
                # Final score - lower is better
                score = energy_factor * energy_weight - distance_factor
                
                # Penalize revisits
                if new_struct in visited_structures:
                    steps_since_visit = iterations - visited_structures[new_struct]
                    revisit_penalty = 10.0 / (1.0 + steps_since_visit/10.0)
                    score += revisit_penalty
                
                # Add to candidates
                candidates.append((score, (new_struct, move_type, (i, j))))
            
            # Sort candidates by score (lower is better)
            candidates.sort(key=lambda x: x[0])
            
            # If we have candidates, select one probabilistically from the top beam_width
            if candidates:
                # Limit to beam width
                top_candidates = candidates[:beam_width]
                
                # Select probabilistically - better scores get higher probability
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
                # Fallback to random if no candidates (shouldn't happen)
                selected_move = random.choice(valid_moves)
        
        # Apply the selected move
        new_struct, move_type, (i, j) = selected_move
        
        # Update current structure
        if move_type == 'add':
            current.add_pair(i-1, j-1)  # Convert from 1-indexed to 0-indexed
        else:  # move_type == 'remove'
            current.remove_pair(i-1, j-1)  # Convert from 1-indexed to 0-indexed
            
        current_db = new_struct
        path.append(RNAStructure(current.sequence, base_pairs=current.base_pairs.copy()))
        
        # Update tabu list (keep it to fixed size)
        tabu_list.add(current_db)
        if len(tabu_list) > tabu_size:
            # Remove oldest element from tabu list
            # For a set, we'll just remove an arbitrary element, which is fine
            tabu_list.remove(next(iter(tabu_list)))
        
        # Update visited structures
        visited_structures[current_db] = iterations
        
        # Check for lack of progress toward target
        new_distance = structure_distance(current_db, end_db)
        if new_distance >= current_distance:
            no_progress_count += 1
        else:
            # Reset counter when making progress
            no_progress_count = 0
        
        # If we've made no progress for too many iterations, increase randomness temporarily
        if no_progress_count > 30:
            # Print a status update if it's taking a long time
            if iterations % 100 == 0:
                print(f"Iteration {iterations}, current distance: {new_distance}, best: {best_distance_so_far}")
            
            # If we're really stuck, try a more direct approach
            if no_progress_count > 100:
                # Try to direct the search more explicitly - identify a base pair in end
                # that's not in current and try to add it (or vice versa)
                current_bp = current.base_pairs
                end_bp = end.base_pairs
                
                # Find base pairs to add or remove
                to_add = end_bp - current_bp
                to_remove = current_bp - end_bp
                
                direct_move_made = False
                
                # First try to remove a pair that's not in the target
                for i, j in to_remove:
                    # See if there's a move to remove this exact pair
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
                        # See if there's a move to add this exact pair
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
                
                # Reset counters if we made a direct move
                if direct_move_made:
                    no_progress_count = 0
            
            # If still stuck, fall back to more randomness
            if not direct_move_made:
                random_chance = min(0.9, random_chance * 2)  # Increase randomness but cap at 90%
                no_progress_count = 0
        else:
            random_chance = max(0.1, random_chance * 0.95)  # Gradually decrease randomness
            
        # Check if we've reached the target
        if current.base_pairs == end.base_pairs:
            break
            
    # If we didn't reach the target, try a direct path from where we are
    if current.base_pairs != end.base_pairs and iterations >= max_iterations:
        print(f"Warning: Indirect path search reached iteration limit ({max_iterations}). "
              f"Final distance: {structure_distance(current_db, end_db)}")
        
        # If we're somewhat close to the target, try a final direct approach
        if structure_distance(current_db, end_db) < initial_distance / 2:
            direct_finish = find_direct_path(current, end)
            # Remove the first element as it's already the last element of our path
            if direct_finish and len(direct_finish) > 1:
                path.extend(direct_finish[1:])
    
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
    
    # Try different parameter combinations for diverse attempts
    for attempt in range(adjusted_attempts):
        print(f"Starting indirect path attempt {attempt+1}/{adjusted_attempts}")
        
        # Vary parameters for diversity
        if attempt == 0:
            # First attempt: balanced approach
            random_chance = 0.2
            tabu_size = 15
        elif attempt == 1:
            # Second attempt: more exploration
            random_chance = 0.4
            tabu_size = 20
        elif attempt == 2:
            # Third attempt: more exploitation
            random_chance = 0.1
            tabu_size = 10
        else:
            # Subsequent attempts: random parameters
            random_chance = 0.1 + (0.4 * random.random())
            tabu_size = random.randint(10, 25)
        
        # For complex structures, increase iterations
        max_iterations = 1000 + bp_distance * 10
        
        try:
            path = find_indirect_path(start, end, fc, 
                                     max_iterations=max_iterations,
                                     tabu_size=tabu_size, 
                                     random_chance=random_chance)
            
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
        print("Lowest Barrier: {lowest_barrier}")
    
    return best_path
