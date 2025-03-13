from typing import List
from src.rna_structure import RNAStructure
import RNA

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

def turner_energy(fold_compound: RNA.RNA.fold_compound, structure: str) -> float:
    """Calculates the Turner Energy of a Structure using the ViennaRNA Package

    Args:
        fold_compound (RNA.RNA.fold_compound): fold compound for an RNA sequence, e.g.: fold_compound = RNA.fold_compound("GCGCGAUACG")
        structure (str): test structure (dot-bracket notation), e.g.: structure = "((....)).."

    Returns:
        float: Turner energy of this structure
    """
    return fold_compound.eval_structure(structure)