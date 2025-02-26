import numpy as np

def calculate_y(H, m):
    """
    Calculates the normal dot product between a bit-packed binary matrix H and a bit-packed binary vector m.

    Both H and m are stored in uint64 units. For each row in H, the dot product is computed by:
      - Performing a bitwise AND between corresponding uint64 units in the row and m.
      - Counting the number of 1 bits in the result (using popcount).
      - Summing these counts to get the dot product.

    Parameters:
      H (ndarray): A matrix with shape (k, num_units) where each row is stored in bit-packed uint64 format.
      m (ndarray): A bit-packed binary vector of shape (num_units,).

    Returns:
      list: A list of integers, where each element is the dot product of the corresponding row of H with m.
    """
    dot_products = []

    # Iterate over each row in H
    for row in H:
        total = 0
        # For each corresponding uint64 unit in the row and m, compute the bitwise AND and count the ones.
        for unit, m_unit in zip(row, m):
            total += bin(unit & m_unit).count("1")
        dot_products.append(total)

    return dot_products