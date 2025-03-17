import numpy as np
from cls_generate import generate, compute_y
from cls_uint64_tools import packed_uint64_length, bitpacked_dot_row_readable, bitpacked_dot_row_optimized, \
    bitpacked_dot_column_optimized, pack2uint64

# Define size of random matrix H (n columns and k rows)
n, k, t = 200, 100, 2

def calculate_m(H, y, t):
    useful_columns = packed_uint64_length(H[0])

    # Calculate Phi: For each column of H, compute dot product with y.
    phi = bitpacked_dot_column_optimized(H, y, useful_columns,True)

    # Create score vector by sorting Phi by value
    score = sorted(phi, key=lambda phi: phi[1], reverse=True)

    # Generate new "m" vector based on Score_sorted
    m = [0] * useful_columns
    for idx, (cnt, val) in enumerate(score[:t]):
        m[cnt] = 1

    return pack2uint64(np.array(m))

def IsSolution(H, y, m):
    y_new = bitpacked_dot_row_optimized(H, m)
    if (y == y_new).all():
        return True
    else:
        return False

if __name__ == "__main__":
    H, m = generate(n, k, t)
    n_H = packed_uint64_length(H)
    n_m = packed_uint64_length(m)
    y_readable = bitpacked_dot_row_readable(H, m)
    y_optimized = bitpacked_dot_row_optimized(H, m)
    y = y_optimized
    m_new = calculate_m(H, y, t)
    y_new = bitpacked_dot_row_optimized(H, m_new)

    if (y == y_new).all():
        print("Yes")
    else:
        print("No")