# Project Outline

## 1. Biological and Algorithmic Background

### 1.1. RNA Secondary Structure
- **What is RNA?**  
  Ribonucleic acid (RNA) is a single-stranded molecule composed of nucleotides: A, U, C, G.
- **What is secondary structure?**  
  Despite being single-stranded, portions of the RNA can fold back and pair up. These **base pairs** are typically A–U or C–G (and sometimes G–U). A secondary structure is a set of these base pairs subject to the rule that they do not "cross."  
  - **Non-crossing constraint**: If (i, j) is a base pair and (k, l) is another base pair, you cannot have `i < k < j < l`.

### 1.2. Energy Models
- **Base pair minimization energy model (simplest approach)**  
  In this model, the "energy" of a structure can (very naively) be considered proportional to the negative of the number of base pairs. The fewer base pairs you have, the higher the "energy," and the more base pairs you have, the lower the "energy." This is not very realistic biologically, but it is a simpler stepping stone.
- **Turner energy model (realistic approach)**  
  The Vienna RNA Package implements the Turner energy rules, which are derived from thermodynamics. These rules take into account loops, stacking, hairpins, bulges, etc. You will be using the function `fold_compound::eval_structure` to get the "real" energy of an RNA secondary structure.

### 1.3. RNA Folding Landscape and Folding Pathways
- **Folding landscape**  
  Imagine all possible valid secondary structures of an RNA sequence arranged in a giant graph, where each structure is a node. Edges connect structures that can be formed by a single base pair change (either adding or removing one base pair).
- **Folding pathway**  
  A sequence \( S_0, S_1, \dots, S_l \) of secondary structures such that:
  1. \( S_0 = L \) (a "left" or initial structure) and \( S_l = R \) (a "right" or target structure).
  2. Each \( S_{i} \) and \( S_{i+1} \) differ by exactly one base pair (an elementary move).
  3. The **barrier** or "bottleneck" in that pathway is the maximum energy encountered along the path (minus the energy of the starting structure, if following the formal definition).

### 1.4. The Barrier Computation Problem
- **Goal**: Find a path from structure \( L \) to structure \( R \) such that this maximum energy "spike" is as low as possible.
- **Why is it hard?**  
  The problem is known to be NP-complete under various models. Since enumerating all paths is (in the worst case) exponential, we resort to heuristics.

## 2. Implementation Overview

### 2.1. Project Structure
- **src/rna_structure.py**: Core RNA structure representation and manipulation
- **src/path_finding.py**: Implementations of direct and indirect path finding algorithms
- **src/energy.py**: Energy calculation functions (base pair minimization and Turner energy models)
- **src/evolution.py**: Evolutionary algorithm for RNA folding pathway search
- **src/utils.py**: Utility functions and helper classes
- **src/evaluation.py**: Functions for evaluating and comparing algorithm performance
- **src/example_rna.py**: Example RNA sequences for testing

### 2.2. Algorithm Implementations

#### 2.2.1. Direct Path Heuristic (direct_heuristic.ipynb)
The direct path heuristic provides a straightforward approach to finding folding pathways by:
- Only considering steps that directly transform the start structure into the end structure
- Prioritizing moves that reduce the structural distance to the target
- Choosing the order of operations that minimizes the energy barrier

This approach is efficient but may miss optimal pathways that require temporary "detours" in the folding landscape.

#### 2.2.2. Indirect Path Heuristic (indirect_heuristic.ipynb)
The indirect path heuristic employs more sophisticated search strategies:
- Uses guided local search to explore the folding landscape
- Implements tabu search to avoid revisiting previously explored structures
- Applies beam selection to focus on the most promising pathways
- Incorporates adaptive randomness to escape local minima

This approach generally finds lower energy barriers than the direct path method, especially for complex RNA structures.

#### 2.2.3. Evolutionary Approach (evolutionary_heuristic.ipynb)
The evolutionary algorithm provides a population-based search strategy:
- Maintains a diverse population of candidate pathways
- Applies selection pressure toward lower energy barriers
- Uses crossover and mutation operations to explore the solution space
- Features configurable parameters for population size, generations, and selection criteria

## 3. Evaluation and Analysis

### 3.1. Performance Metrics (evaluation.ipynb)
The algorithms are evaluated on the following metrics:
- **Energy barrier height**: The maximum energy encountered along the folding pathway
- **Path length**: The number of elementary moves in the folding pathway
- **Execution time**: Computational efficiency of the algorithm

### 3.2. Comparative Analysis (comparisons.ipynb)
Based on our experimental results:
- The direct path heuristic provides the fastest solution but often with higher energy barriers
- The indirect path heuristic typically finds lower energy barriers at a moderate computational cost
- The evolutionary approach can discover the lowest barriers for many sequences but has the highest computational cost

### 3.3. Complexity Analysis (complexity_analysis.ipynb)
Theoretical time and space complexity:
- Direct path: O(n²) time where n is the sequence length
- Indirect path: O(b·n³) time where b is the beam width
- Evolutionary approach: O(p·g·n³) time where p is population size and g is number of generations

Empirical evaluation confirms these complexity bounds and demonstrates the trade-off between solution quality and computational resources.

## 4. Usage Examples

### 4.1. Basic Usage
```python
from src.rna_structure import RNAStructure
from src.path_finding import find_direct_path, find_best_indirect_path
from src.energy import turner_energy

# Define RNA sequence and structures
sequence = "GGGAAACCC"
start_structure = "(((...)))"
end_structure = "...(......"

# Create RNA Structure Objects
start_structure = rs.RNAStructure(seq, structure=start_struct)
end_structure = rs.RNAStructure(seq, structure=end_struct)

# Get direct path
direct_path = pf.find_direct_path(start_structure, end_structure)
direct_path = [struct.to_dotbracket() for struct in direct_path]

# Get indirect path
indirect_path = pf.find_best_indirect_path(start_structure, end_structure)
indirect_path = [struct.to_dotbracket() for struct in indirect_path]

# Get evolutionary path
best, steps = ev.best_folding(seq, start_struct, end_struct, N=200, max_steps=50)
evolutionary_path = best['path']
```

## 5. Future Work

### 5.1. Algorithmic Improvements
- Implement advanced search techniques like A* with admissible heuristics
- Explore reinforcement learning approaches for path finding
- Develop methods for handling pseudoknots in RNA structures

### 5.2. Biological Applications
- Apply energy barrier analysis to RNA design problems
- Investigate co-transcriptional folding pathways
- Study the relationship between energy barriers and RNA function