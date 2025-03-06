from typing import List, Tuple, Set
from rna_structure import RNAStructure

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