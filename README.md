# rna_energy_barriers
Determine energy barriers in RNA folding landscapes

# Project Outline

## 1. Biological and Algorithmic Background

### 1.1. RNA Secondary Structure
- **What is RNA?**  
  Ribonucleic acid (RNA) is a single-stranded molecule composed of nucleotides: A, U, C, G.
- **What is secondary structure?**  
  Despite being single-stranded, portions of the RNA can fold back and pair up. These **base pairs** are typically A–U or C–G (and sometimes G–U). A secondary structure is a set of these base pairs subject to the rule that they do not “cross.”  
  - **Non-crossing constraint**: If (i, j) is a base pair and (k, l) is another base pair, you cannot have `i < k < j < l`.

### 1.2. Energy Models
- **Base pair minimization energy model (simplest approach)**  
  In this model, the “energy” of a structure can (very naively) be considered proportional to the negative of the number of base pairs. The fewer base pairs you have, the higher the “energy,” and the more base pairs you have, the lower the “energy.” This is not very realistic biologically, but it is a simpler stepping stone.
- **Turner energy model (realistic approach)**  
  The Vienna RNA Package implements the Turner energy rules, which are derived from thermodynamics. These rules take into account loops, stacking, hairpins, bulges, etc. You will be using the function `fold_compound::eval_structure` to get the “real” energy of an RNA secondary structure.

### 1.3. RNA Folding Landscape and Folding Pathways
- **Folding landscape**  
  Imagine all possible valid secondary structures of an RNA sequence arranged in a giant graph, where each structure is a node. Edges connect structures that can be formed by a single base pair change (either adding or removing one base pair).
- **Folding pathway**  
  A sequence \( S_0, S_1, \dots, S_l \) of secondary structures such that:
  1. \( S_0 = L \) (a “left” or initial structure) and \( S_l = R \) (a “right” or target structure).
  2. Each \( S_{i} \) and \( S_{i+1} \) differ by exactly one base pair (an elementary move).
  3. The **barrier** or “bottleneck” in that pathway is the maximum energy encountered along the path (minus the energy of the starting structure, if following the formal definition).

### 1.4. The Barrier Computation Problem
- **Goal**: Find a path from structure \( L \) to structure \( R \) such that this maximum energy “spike” is as low as possible.
- **Why is it hard?**  
  The problem is known to be NP-complete under various models. Since enumerating all paths is (in the worst case) exponential, we resort to heuristics.

---

## 2. Core Requirements (Direct vs. Indirect Paths)

### 2.1. Direct Path Heuristic
- **Idea**: Only remove base pairs that are in \( L \) but not in \( R \), and only add base pairs that are in \( R \) but not in \( L \). You do this in some order:
  1. **Remove** unneeded base pairs from \( L \) (i.e., those that are not in \( R \)).
  2. **Add** the new base pairs that appear in \( R \).

  At each intermediate structure, you measure the energy, and the highest energy structure along this route is your barrier for that particular direct path.

- **Why direct?**  
  Because at every step, your distance to \( L \) is strictly increasing (by removing base pairs that are in \( L \)) and your distance to \( R \) is strictly decreasing (by adding base pairs that are in \( R \)). You do **not** allow any “extra” base pairs that aren’t in \( L \) or \( R \) in between. Conceptually, there is only one set of pairs: either remove them (if they’re not in \( R \)) or add them (if they’re missing in \( L \) but present in \( R \)).

### 2.2. Indirect Path Heuristic
- **Idea**: Sometimes, a direct path is not optimal. Maybe you want to add (or remove) an intermediate base pair that does not appear in \( L \) nor in \( R \) but temporarily **lowers** the energy on the way. That can avoid a huge energy “spike.” 
- **Heuristic**:  
  1. From the current structure, look at all possible single base-pair additions/removals that keep you in a valid structure.  
  2. Evaluate their energies using `fold_compound::eval_structure`.  
  3. Choose the “best” next move (for example, the move that leads to the smallest rise in energy, or the largest drop in energy, etc.).  
  4. Continue until you reach \( R \).  

  You can do more advanced heuristics too, like keeping a small “frontier” of the k-best next structures at each step instead of a single best.

---

## 3. Complexity and NP-Completeness

### 3.1. Why is it so Complex?
- The space of possible RNA secondary structures for a length \( n \) is huge (often super-polynomial).  
- Each structure can be connected to many neighbors (in principle, on the order of \( n^2 \) possible base pairs to add/remove).
- Doing a naive BFS or DFS in this space is typically exponential for large \( n \).

### 3.2. Proposed Bound of Complexities
- **Direct Path**:  
  - Worst-case path length: Up to the total difference in base pairs between \( L \) and \( R \). If each has up to \( O(n) \) base pairs, the worst-case path can be \( O(n) \).  
  - But you might need to re-check energy after each base pair change. That might be \( O(n) \) or \( O(n^2) \) inside Vienna’s engine per structure. 
  - So the direct path approach might be polynomial in \( n \) but only because it is ignoring a huge part of the landscape (it’s not searching all possible intermediate states).

- **Indirect Path**:
  - If you do a local greedy approach, at each step you might explore \( O(n^2) \) possible single base pair changes. Then you pick one and move on. So you might have something like \( O(n^3) \) in a naive worst case. 
  - If you keep a “beam” (k-best frontier) at each step, it becomes bigger: each frontier can have up to \( k \) structures, each can lead to \( O(n^2) \) neighbors, etc. That can blow up quickly.

The professor just wants you to reason that “the search space is large, so the naive approach is exponential,” or give partial polynomial/“exponential in the worst case” arguments.

---

## 4. How to Start Implementing

### 4.1. Parsing and Representing RNA Structures
- **Sequence**: You will have your RNA string, e.g. `"GCAUUC..."`.
- **Structure**: Usually represented in dot-bracket notation (e.g., `"((..))..."`), or equivalently as a set of pairs `\{(i, j), ...\}`.  
  - For the direct path approach, you can easily store which base pairs are in \( L \), which are in \( R \), and figure out the difference sets.

### 4.2. Computing the Energy (ViennaRNA)
- **fold_compound::eval_structure**:  
  1. Construct a fold compound for your sequence using ViennaRNA’s API.  
  2. Use `eval_structure(structure_string)` or `eval_structure(list_of_pairs)` to get the Turner energy.

### 4.3. Implementing a Direct Path Heuristic
1. **Determine which pairs are in \( L \) but not \( R \)**. Call this set \( L - R \).  
2. **Determine which pairs are in \( R \) but not \( L \)**. Call this set \( R - L \).  
3. **Compute an order in which you remove pairs from \( L - R \)**. Possibly sorted by some criterion (like removing the highest-energy base pair first or just in the order they appear).  
4. **After all pairs from \( L - R \) are removed, add pairs from \( R - L \)**. Possibly in some order.  
5. **At each step** (i.e., after each addition/removal), call `eval_structure` to get the energy. Track the maximum energy on the path. That is your barrier for that path.

You can experiment with different orderings (which pair you remove first, etc.) to see if you get a lower barrier.

### 4.4. Implementing an Indirect Path Heuristic
1. **Start from \( L \)**.  
2. **At each structure** \( S \), consider all valid single base pair additions or removals (not just those that lead you directly closer to \( R \) in terms of set difference).  
3. **Compute the energy** of each neighbor using `eval_structure`.  
4. **Pick the neighbor** that gives the best improvement (or the smallest penalty) and move there.  
5. **Repeat** until you reach \( R \).  
   - If you never reach \( R \) (because of local minima), you may need to get more sophisticated (like taboo moves, BFS, or best-first search with a priority queue, etc.).

The above is a basic hill-climbing or greedy approach; you could also do:
- **Beam search**: keep the top k next structures.  
- **Depth-limited search**: keep exploring until the path gets too long.  
- **Best-first search** (like A*): define a cost function that depends on how many pairs remain different from \( R \) plus the energy barrier so far.

---

## 5. Study Results on Examples

### 5.1. Generate or Obtain Example RNA Sequences
- You can take short sequences (length ~ 10–30) to keep it manageable.
- For each sequence, define:
  1. A “left” structure \( L \).
  2. A “right” structure \( R \).
- Or you can generate random structures using an external tool or just define them by hand for small examples.

### 5.2. Compare Direct vs. Indirect
- **Direct path approach**: 
  - Note the order you remove/add pairs. 
  - Compute the maximum energy. 
  - That’s your direct path barrier.
- **Indirect path approach**:
  - Possibly find a path with a lower barrier. 
  - Compare how much smaller your barrier is relative to direct path.

### 5.3. Discuss Failure Cases
- Show examples where the direct path leads to a big energy spike, but the indirect path finds a smaller barrier.
- Conversely, you might find examples where direct path is already optimal.

---

## 6. Putting It All Together

1. **Implementation**  
   - Use a programming language of your choice (Python, C++, etc.).  
   - Integrate or wrap the ViennaRNA library (they have APIs for Python, C, C++, etc.).  
   - Create functions:
     - `eval_energy(sequence, structure)` -> returns energy.
     - `neighbors(structure)` -> returns all single base pair addition/removal neighbors (valid structures).
     - `direct_path(L, R)` -> returns a path and barrier.
     - `indirect_path(L, R)` -> returns a path and barrier.

2. **Complexity Analysis**  
   - The search space is exponential in the worst case.  
   - But direct path is linear in the difference in base pairs.  
   - Indirect path can blow up, so a purely naive approach can be exponential. You rely on heuristics.

3. **Experiment**  
   - Use small test sequences so that your code runs in a reasonable time.  
   - Show tables of results:  
     - Sequence length, difference in base pairs, direct path barrier, indirect path barrier, runtime, etc.

4. **Report**  
   - Summarize your approach, complexities, why it’s NP-hard, how your heuristics approximate solutions.  
   - Show interesting examples with lower/higher barriers.  
   - Possibly mention advanced heuristics if you have time (like Morgan–Higgs, BFS variants, or beam search).

---

## 7. Tips for Getting Started

1. **Start with the Base Pair Minimization Model**  
   - You can avoid the complexity of the Turner energy for an initial test. Just treat the energy as \(-\text{(number_of_pairs)}\), or something extremely simple.  
   - Implement your direct/indirect path code and debug the logic for enumerating neighbors.

2. **Add ViennaRNA**  
   - Once you have your logic stable, switch from the toy “count base pairs” energy model to the real Turner energies via `eval_structure`.  
   - Make sure you have correct installation and can call the `eval_structure` function.

3. **Plan How to Represent Structures Internally**  
   - A straightforward approach is to keep them as sets of pairs (i,j).  
   - Use a function to convert a set of pairs to dot-bracket notation if needed by ViennaRNA.

4. **Document and Test**  
   - Write test cases for very short sequences (e.g., length 4 or 5) where you can manually compute all structures to confirm correctness.

---

## Final Words

- **Main Challenge**: This project is less about advanced biology and more about graph exploration with a complex energy function.
- **Don’t Panic Over Biology**: You only need enough biology to know how to interpret RNA base pairs and how to use the Turner energy function.  
- **Focus** on the heuristic search approach, data structures, and complexity analysis.
