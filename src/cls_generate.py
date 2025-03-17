import math
import numpy as np
from cls_uint64_tools import pack2uint64

def extract_t(m):
    return np.unpackbits(m.view(np.uint8)).sum() - 1

def compute_y(H, m):
    """
    Computes the y = H dot m.
    Returns y as a list of integers.
    """
    y = H.dot(m)
    return y.tolist()

def generate_H(n, k):
    """
    Generates a random binary parity-check matrix H with n columns and k rows,
    and returns it in bit-packed uint64 format with an extra sentinel bit 1 in
    each row, which marks the end of the row and allows reconstruction of n.
    The matrix is regenerated if it has no full row rank before packing into uint64 units.

    Parameters:
    n (int): Number of columns (without sentinel bit)
    k (int): Number of rows

    Returns:
    H (ndarray): The bit-packed uint64 matrix with sentinel bits encoded in each row.
    """

    def rank_mod2(matrix):
        """Computes the rank of a binary matrix over GF(2) using Gaussian elimination."""
        A = matrix.copy() % 2
        rows, cols = A.shape
        rank = 0
        for col in range(cols):
            pivot = -1
            for row in range(rank, rows):
                if A[row, col] == 1:
                    pivot = row
                    break
            if pivot == -1:
                continue
            A[[rank, pivot]] = A[[pivot, rank]]
            for r in range(rows):
                if r != rank and A[r, col] == 1:
                    A[r] = (A[r] + A[rank]) % 2
            rank += 1
        return rank

    while True:
        # Generate random binary elements of the matrix
        H_binary = np.random.randint(0, 2, (k, n), dtype=np.uint8)
        # Ensure full row rank before continuing
        if rank_mod2(H_binary) == k:
            break

    return pack2uint64(H_binary)


def generate_m(n, t):
    """
    Generates a random binary vector m with Hamming weight t and returns it in bit-packed
    uint64 format with an extra sentinel bit 1 added to end, which marks it's end and allows
    reconstruction of n.

    Parameters:
    n (int): Length of the vector (useful length)
    t (int): Hamming weight (number of ones in the vector)

    Returns:
    m (ndarray): The bit-packed uint64 vector with sentinel bit encoded.
    """

    # Create m with t ones followed by (n-t) zeros and shuffle
    m_binary = np.concatenate((np.ones(t, dtype=np.uint8), np.zeros(n - t, dtype=np.uint8)))
    np.random.shuffle(m_binary)

    return pack2uint64(m_binary)


def generate(n, k, t):
    """
    Generates a parity-check matrix H and a binary vector m.

    Parameters:
    n (int): Number of columns of H (and length of m)
    k (int): Number of rows of H
    t (int): Hamming weight of m

    Returns:
    tuple: (H, m) where H is the generated matrix and m is the generated vector
    """

    H = generate_H(n, k)
    m = generate_m(n, t)

    return H, m
