import numpy as np
import RNA
import matplotlib.pyplot as plt

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

_complementary_bases = set(["AU","CG","GC","GU","UA","UG"])
def complementary(x,y):
    """
    Test complementarity of two bases
    Args:
        x: first base
        y: second base
    Returns: True, if complementary
    """
    return x+y in _complementary_bases

def nussinov_base_pair_maximization_unambiguous(sequence, *, m=3):
    """
    Maxmimize the number of base pairs in any non-crossing structure of an RNA
    
    Args:
        sequence: the RNA sequence
        m: the minimum loop length, i.e. for all base pairs: i<j-m
    Return:
        the dynamic programming matrix F,
        where F_ij is the max number of base pairs for the subsequence i..j

    Note: implementes an unambiguous recursion equation
    """
    n = len(sequence)
    F = np.zeros((n,n), dtype=int)

    # initialization (here, this does nothing, since the matrix is already filled by zeros)
    for i in range(n):
        for j in range(i, min(n,i+m+1)):
            F[i,j] = 0

    # fill non-init entries due to recursion equation
    for i in reversed(range(n)):
        for j in range(i+m+1,n):
            F[i,j] = F[i+1,j]
            
            if complementary(sequence[i],sequence[j]):
                F[i,j] = max(F[i,j], F[i+1,j-1] + 1)

            for k in range(i+m+1,j):
                if complementary(sequence[i],sequence[k]):  # Changed from sequence[j] to sequence[k]
                    F[i,j] = max(F[i,j], F[i+1,k-1] + 1 + F[k+1,j])
            
    return F

# example
# nussinov_base_pair_maximization_unambiguous("GGGGAACAAUUCC", m=3)

def bplist2db(n, basepairs):
    """
    Convert list of base pairs to dot bracket notation
    
    Args:
        n: sequence length
        basepairs: list of base pairs
    Returns:
        dot bracket string
    """
    db = ['.']*n
    for (i,j) in basepairs:
        db[i], db[j] = '(', ')'
    return "".join(db)

# example
#bplist2db(20,[(0,19),(1,17),(2,8),(3,7),(9,16),(10,14)])

def dotbracket_to_pt(structure):
    """
    Convert a dot-bracket string to a pair table.
    Returns a list 'pt' of length n+1 (1-indexed; pt[0] is unused).
    For each i (1-indexed), pt[i] = j if base i is paired with j, else 0.
    """
    n = len(structure)
    pt = [0]*(n+1)
    stack = []
    for i, char in enumerate(structure, start=1):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pt[i] = j
                pt[j] = i
    return pt

def plot_energy_profiles(seq, evolutionary_path, direct_path, indirect_path):
    """
    Plot energy profiles for both evolutionary and direct paths
    
    Args:
        seq: RNA sequence
        evolutionary_path: List of structures from evolutionary algorithm
        direct_path: List of structures from direct path
        indirect_path: List of structures from indirect path

    """
    fc = RNA.fold_compound(seq)
    
    # Calculate energies for evolutionary path
    evo_energies = [fc.eval_structure(struct) for struct in evolutionary_path]
    evo_steps = range(len(evo_energies))
    
    # Calculate energies for direct path
    direct_energies = [fc.eval_structure(struct) for struct in direct_path]
    direct_steps = range(len(direct_energies))

    # Calculate energies for indirect path
    indirect_energies = [fc.eval_structure(struct) for struct in indirect_path]
    indirect_steps = range(len(indirect_energies))
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(evo_steps, evo_energies, color='#55a868', marker='o', label='Evolutionary Path')
    plt.plot(direct_steps, direct_energies, color='#4c72b0', marker='o', label='Direct Path')
    plt.plot(indirect_steps, indirect_energies, color='#dd8452', marker='o', label='Indirect Path')

    
    # Add labels and title
    plt.xlabel('Step')
    plt.ylabel('Energy (kcal/mol)')
    plt.title('Energy Profiles Comparison')
    plt.grid(True)
    plt.legend()
    
    # Show the plot
    plt.show()
    
    # Print some statistics
    print(f"Evolutionary Path:")
    print(f"- Initial energy: {evo_energies[0]:.2f} kcal/mol")
    print(f"- Final energy: {evo_energies[-1]:.2f} kcal/mol")
    print(f"- Maximum energy: {max(evo_energies):.2f} kcal/mol")
    print(f"- Energy barrier: {max(evo_energies) - evo_energies[0]:.2f} kcal/mol")
    print(f"- Path length: {len(evo_energies)} steps")
    
    print(f"\nDirect Path:")
    print(f"- Initial energy: {direct_energies[0]:.2f} kcal/mol")
    print(f"- Final energy: {direct_energies[-1]:.2f} kcal/mol")
    print(f"- Maximum energy: {max(direct_energies):.2f} kcal/mol")
    print(f"- Energy barrier: {max(direct_energies) - direct_energies[0]:.2f} kcal/mol")
    print(f"- Path length: {len(direct_energies)} steps")

    print(f"\nIndirect Path:")
    print(f"- Initial energy: {indirect_energies[0]:.2f} kcal/mol")
    print(f"- Final energy: {indirect_energies[-1]:.2f} kcal/mol")
    print(f"- Maximum energy: {max(indirect_energies):.2f} kcal/mol")
    print(f"- Energy barrier: {max(indirect_energies) - indirect_energies[0]:.2f} kcal/mol")
    print(f"- Path length: {len(indirect_energies)} steps")