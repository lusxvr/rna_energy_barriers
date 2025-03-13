import RNA
import src.path_finding as pf
import src.evolution as ev
import src.rna_structure as rs

def direct_path_algorithm(seq, start_struct, end_struct):
    """Wrapper for direct path finding algorithm.
    
    Args:
        seq: RNA sequence
        start_struct: Starting structure in dot-bracket notation
        end_struct: Target structure in dot-bracket notation
        
    Returns:
        tuple: (path, energy_profile, barrier)
            - path: List of structures in dot-bracket notation
            - energy_profile: List of energies for each structure
            - barrier: Maximum energy barrier encountered
    """
    # Convert to RNAStructure objects
    start_structure = rs.RNAStructure(seq, structure=start_struct)
    end_structure = rs.RNAStructure(seq, structure=end_struct)
    
    # Get direct path
    direct_path_objects = pf.find_direct_path(start_structure, end_structure)
    path = [struct.to_dotbracket() for struct in direct_path_objects]
    
    # Calculate energy profile
    fc = RNA.fold_compound(seq)
    energy_profile = [fc.eval_structure(struct) for struct in path]
    
    # Calculate energy barrier
    start_energy = energy_profile[0]
    barrier = max([energy - start_energy for energy in energy_profile])
    
    return path, energy_profile, barrier

def evolutionary_algorithm(seq, start_struct, end_struct, N=100, max_steps=100, alpha=0.7, T=pf.T_CONST, beta=0):
    """Wrapper for evolutionary path finding algorithm.
    
    Args:
        seq: RNA sequence
        start_struct: Starting structure in dot-bracket notation
        end_struct: Target structure in dot-bracket notation
        N: Population size
        max_steps: Maximum number of evolution steps
        
    Returns:
        tuple: (path, energy_profile, barrier)
            - path: List of structures in dot-bracket notation
            - energy_profile: List of energies for each structure
            - barrier: Maximum energy barrier encountered
    """
    # Run evolutionary algorithm
    best_result, steps = ev.best_folding(seq, start_struct, end_struct, 
                                        N=N, max_steps=max_steps,
                                        alpha=alpha, T=T, beta=beta)
    
    # Extract path and energy profile
    path = best_result['path']
    energy_profile = best_result['energy_profile']
    
    # Calculate barrier (or use the one already calculated)
    barrier = best_result['highest_energy']
    
    return path, energy_profile, barrier

def indirect_path_algorithm(seq, start_struct, end_struct):
    """Wrapper for indirect path finding algorithm.
    
    Args:
        seq: RNA sequence
        start_struct: Starting structure in dot-bracket notation
        end_struct: Target structure in dot-bracket notation
        
    Returns:
        tuple: (path, energy_profile, barrier)
            - path: List of structures in dot-bracket notation
            - energy_profile: List of energies for each structure
            - barrier: Maximum energy barrier encountered
    """
    # This is a placeholder - replace with your actual indirect path implementation
    # For example, you might have a function in a module like:
    # indirect_path = indirect_path_module.find_indirect_path(start_struct, end_struct, seq)
    
    # Create fold compound for energy calculations
    fc = RNA.fold_compound(seq)
    
    # Convert to RNAStructure objects for compatibility with your existing code
    start_structure = rs.RNAStructure(seq, structure=start_struct)
    end_structure = rs.RNAStructure(seq, structure=end_struct)
    
    # Call your indirect path finding function (replace with actual implementation)
    # For now, I'll just import it assuming it exists in your path_finding module
    indirect_path_objects = pf.find_indirect_path(start_structure, end_structure)
    path = [struct.to_dotbracket() for struct in indirect_path_objects]
    
    # Calculate energy profile
    energy_profile = [fc.eval_structure(struct) for struct in path]
    
    # Calculate energy barrier
    start_energy = energy_profile[0]
    barrier = max([energy - start_energy for energy in energy_profile])
    
    return path, energy_profile, barrier