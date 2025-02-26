import math

import numpy as np


def generate_H(n, k, cnt):
    """
    Generates a random binary parity-check matrix H with n columns and k rows, stored in bit-packed
    uint64 format. An extra 1 is added to each row, which marks it's end and allows reconstruction of n.
    The matrix is regenerated if it has no full row rank before packing into uint64 units.

    Parameters:
    n (int): Number of columns (useful columns)
    k (int): Number of rows

    Returns:
    H (ndarray): The bit-packed matrix with n+1 encoded.
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

    # Number of uint64 units per row (includes n columns, extra '1' bit and padded zeros up to multiple of 64).
    num_units = math.ceil((n + 1) / 64)

    while True:
        cnt[0] = cnt[0] + 1
        # Generate random binary elements of the matrix
        H_binary = np.random.randint(0, 2, (k, n), dtype=np.uint8)
        # Ensure full row rank before continuing
        if rank_mod2(H_binary) == k:
            break

    H = np.zeros((k, num_units), dtype=np.uint64)

    for i in range(k):
        # Append the sentinel '1' at the end
        row_bits = np.append(H_binary[i], 1)

        # Pack into uint64 units
        for j in range(num_units):
            start, end = j * 64, min((j + 1) * 64, len(row_bits))
            bit_segment = row_bits[start:end]

            if bit_segment.size > 0:
                H[i, j] = np.packbits(np.pad(bit_segment, (0, 64 - len(bit_segment)), constant_values=0),
                                      bitorder='little').view(np.uint64)[0]

    return H


def generate_m(n, t):
    """
    Generates a random binary vector m with Hamming weight t, stored in bit-packed uint64 format.
    An extra 1 is added to each row, which marks it's end and allows reconstruction of n.

    Parameters:
    n (int): Length of the vector (useful length)
    t (int): Hamming weight (number of ones in the vector)

    Returns:
    m (ndarray): The bit-packed vector with n+1 encoded.
    """

    # Number of uint64 units per row (includes n columns, extra '1' bit and padded zeros up to multiple of 64).
    num_units = math.ceil((n + 1) / 64)

    # Create m with t ones followed by (n-t) zeros and shuffle
    m_binary = np.concatenate((np.ones(t, dtype=np.uint8), np.zeros(n - t, dtype=np.uint8)))
    np.random.shuffle(m_binary)

    # Append the extra '1' bit at the end
    m_binary = np.append(m_binary, 1)

    # Pack into uint64 units
    m = np.zeros(num_units, dtype=np.uint64)
    for j in range(num_units):
        start, end = j * 64, min((j + 1) * 64, len(m_binary))
        bit_segment = m_binary[start:end]

        if bit_segment.size > 0:
            m[j] = np.packbits(np.pad(bit_segment, (0, 64 - len(bit_segment)), constant_values=0),
                               bitorder='little').view(np.uint64)[0]

    return m


def generate(n, k, t, cnt):
    """
    Generates a parity-check matrix H and a binary vector m.

    Parameters:
    n (int): Number of columns of H (and length of m)
    k (int): Number of rows of H
    t (int): Hamming weight of m

    Returns:
    tuple: (H, m) where H is the generated matrix and m is the generated vector
    """

    H = generate_H(n, k, cnt)
    m = generate_m(n, t)

    return H, m
