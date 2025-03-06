from typing import List, Set, Tuple, Optional
import numpy as np
from utils import complementary, bplist2db, bcolors

class RNAStructure:
    def __init__(self, sequence: str, structure: Optional[str] = None, base_pairs: Optional[Set[Tuple[int, int]]] = None):
        """
        Initialize an RNA structure
        Args:
            sequence: RNA sequence string
            structure: Dot-bracket notation of structure (optional)
            base_pairs: Set of base pairs as (i,j) tuples (optional)
        """
        self.sequence = sequence
        self.length = len(sequence)
        
        if structure is not None:
            self.base_pairs = self._db_to_bplist(structure)
        elif base_pairs is not None:
            self.base_pairs = base_pairs
        else:
            self.base_pairs = set()
            
    def _db_to_bplist(self, db: str) -> Set[Tuple[int, int]]:
        """Convert dot-bracket notation to base pair list"""
        if len(db) != self.length:
            raise ValueError("Structure length doesn't match sequence length")
            
        pairs = set()
        stack = []
        
        for i, char in enumerate(db):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if not stack:
                    raise ValueError("Invalid structure: unmatched closing bracket")
                j = stack.pop()
                pairs.add((j, i))
                
        if stack:
            raise ValueError("Invalid structure: unmatched opening bracket")
            
        return pairs
    
    def to_dotbracket(self) -> str:
        """Convert structure to dot-bracket notation"""
        return bplist2db(self.length, self.base_pairs)
    
    def is_valid_pair(self, i: int, j: int) -> bool:
        """Check if (i,j) would form a valid base pair"""
        # Check indices
        if not (0 <= i < j < self.length):
            print(f"{bcolors.WARNING}WARNING: Invalid indices: i={i}, j={j}, length={self.length}. Some algorithms will propably fail{bcolors.ENDC}")
            return False
        
        # Check complementarity
        if not complementary(self.sequence[i], self.sequence[j]):
            print(f"{bcolors.WARNING}WARNING: Non-complementary bases: {self.sequence[i]}-{self.sequence[j]}. Some algorithms will propably fail{bcolors.ENDC}")
            return False
        
        # Check for conflicts with existing pairs
        for x, y in self.base_pairs:
            # No base can be paired twice
            if i in (x, y) or j in (x, y):
                print(f"{bcolors.WARNING}WARNING: Base {i if i in (x,y) else j} is already paired in pair ({x},{y}). Some algorithms will propably fail{bcolors.ENDC}")
                return False
            # No crossing pairs
            if (x < i < y < j) or (i < x < j < y):
                print(f"{bcolors.WARNING}WARNING: Crossing pairs: trying to add ({i},{j}) would cross existing pair ({x},{y}). Some algorithms will propably fail{bcolors.ENDC}")
                return False
                
        return True
    
    def add_pair(self, i: int, j: int) -> bool:
        """
        Add a base pair if valid
        Returns: True if pair was added, False otherwise
        """
        if self.is_valid_pair(i, j):
            self.base_pairs.add((i, j))
            return True
        return False
    
    def remove_pair(self, i: int, j: int) -> bool:
        """
        Remove a base pair if it exists
        Returns: True if pair was removed, False otherwise
        """
        pair = (i, j)
        if pair in self.base_pairs:
            self.base_pairs.remove(pair)
            return True
        return False
    
    def get_neighbors(self) -> List['RNAStructure']:
        """Get all structures that differ by one base pair"""
        neighbors = []
        
        # Try adding each possible pair
        for i in range(self.length):
            for j in range(i + 1, self.length):
                if self.is_valid_pair(i, j):
                    new_pairs = self.base_pairs | {(i, j)}
                    neighbors.append(RNAStructure(self.sequence, base_pairs=new_pairs))
                    
        # Try removing each existing pair
        for i, j in self.base_pairs:
            new_pairs = self.base_pairs - {(i, j)}
            neighbors.append(RNAStructure(self.sequence, base_pairs=new_pairs))
            
        return neighbors
    
    def base_pair_distance(self, other: 'RNAStructure') -> int:
        """Compute base pair distance between two structures"""
        return len(self.base_pairs ^ other.base_pairs)  # symmetric difference 