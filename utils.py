import numpy as np

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