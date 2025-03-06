from typing import List
from rna_structure import RNAStructure

def base_pair_minimization_energy(structure: RNAStructure) -> float:
    """
    Calculate a simple energy function that is proportional to the negative number of base pairs.
    This implements a basic energy model where each base pair contributes -1 energy unit.
    
    Args:
        structure: An RNA structure object
        
    Returns:
        The negative number of base pairs as a float value (more negative = more stable)
    """
    return -len(structure.base_pairs)