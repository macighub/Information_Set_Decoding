import numpy as np


def rank_mod2(matrix):
    """
    Computes the rank of a binary matrix over GF(2) using Gaussian elimination.
    """
    A = matrix.copy() % 2
    rows, cols = A.shape
    rank = 0
    for col in range(cols):
        pivot = -1
        # Find a pivot in the current column (starting from row "rank")
        for row in range(rank, rows):
            if A[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue  # No pivot in this column, move to next
        # Swap the pivot row with the current row
        A[[rank, pivot]] = A[[pivot, rank]]
        # Eliminate all other ones in this column
        for r in range(rows):
            if r != rank and A[r, col] == 1:
                A[r] = (A[r] + A[rank]) % 2
        rank += 1
    return rank


def generate_random_H(k=50, n=100):
    """
    Generates a random binary parity-check matrix H of shape (k, n)
    with full row rank (rank = k) and then converts each row into an integer.

    Each row is interpreted as a binary number (with the first column being the most significant bit)
    and then converted to its decimal integer equivalent.

    Returns:
      - H_int: A list of k integers, where each integer represents a row of the parity-check matrix.
    """
    # Generate random matrices until one has full row rank mod 2.
    while True:
        H = np.random.randint(0, 2, size=(k, n))
        if rank_mod2(H) == k:
            break

    return H

def generate_random_m(n,t=15):

    """
    Generates a random binary vector m of length n.
    """
    m = ([1] * t) + ([0] * (n-t))
    np.random.shuffle(m)

    return m

    #return np.random.randint(0, 2, size=(n,))


def compute_y(H, m):
    """
    Computes the y = H dot m.
    Returns y as a list of integers.
    """
    y = H.dot(m)
    return y.tolist()

def load_matrix_and_m(filename):
    """
    Loads a parity–check matrix H and a binary vector m from a file.

    File format:
      - Every nonempty line except the last is interpreted as a row of H.
        Each such row is a sequence of binary digits (0's and 1's) written
        without any separators (any non '0'/'1' characters are ignored).
      - The last nonempty line in the file contains the binary digits for m.
      - All rows of H are padded (with zeros) to the maximum row length.
      - If the number of digits in m is less than the number of columns of H,
        zeros are appended to m.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Remove leading/trailing whitespace and ignore empty lines.
    lines = [line.strip() for line in lines if line.strip() != '']
    if len(lines) < 3:
        raise ValueError("File must contain at least 3 nonempty lines: 2 or more rows for H and 1 row for m.")

    # The last line is for m; all previous lines are for the matrix.
    matrix_lines = lines[:-1]
    m_line = lines[-1]

    # Process matrix rows: keep only '0' and '1' characters.
    H_rows = []
    max_len = 0
    for line in matrix_lines:
        # Filter only '0' and '1'
        row_digits = [ch for ch in line if ch in ('0', '1')]
        max_len = max(max_len, len(row_digits))
        H_rows.append(row_digits)

    # Pad each row to have exactly max_len digits.
    H_processed = []
    for row in H_rows:
        row_ints = [int(x) for x in row]
        if len(row_ints) < max_len:
            row_ints.extend([0] * (max_len - len(row_ints)))
        else:
            row_ints = row_ints[:max_len]
        H_processed.append(row_ints)

    H = np.array(H_processed, dtype=int)

    # Process the m line: extract binary digits.
    m_digits = [ch for ch in m_line if ch in ('0', '1')]
    m_ints = [int(x) for x in m_digits]
    # If m has fewer entries than the number of columns of H, pad with zeros.
    if len(m_ints) < max_len:
        m_ints.extend([0] * (max_len - len(m_ints)))
    else:
        m_ints = m_ints[:max_len]

    m = np.array(m_ints, dtype=int)

    return H, m

def generate(k=50, n=100, t=5):
    # Ask the user whether to generate H and m or load them from file.
    print(f"\n{'=' * 32}")
    print("\nEnter:\n")
    print("  T - to load from test input")
    print("  L - to load last used input")
    print("  G - to generate random input.")
    choice = input("\nYour choice: ").strip().upper()

    print("\n")

    if choice == 'G':
        print("Generating H and m")
        #k, n, t = 50, 100, 5 ********(moved to arguments)
        H = generate_random_H(k, n)
        #t = 5 ********(moved to arguments)
        m = generate_random_m(n,t)
    else:
        if choice == 'T':
            print("Loading test input")
            filename = "input_test.dat"
        elif choice == 'L':
            print("Loading last input")
            filename = "input.dat"
        else:
            print("Invalid choice. Please run the program again and choose 'T', 'L' or 'G'.")
            return False

        try:
            H, m = load_matrix_and_m(filename)
        except Exception as e:
            print("Error loading file:", e)
            return False

    print("\nParity matrix:")
    leftText = "H[k,n] = "
    for H_row in H:
        print(f"{leftText}{H_row}")
        leftText = "         "

    print("\nInput vector:")
    print(f"m[n] = {np.array(m)}")

    # Write loaded/generated random H and m to "input.dat"
    # For each row of H, write the concatenated binary digits.
    # Then write an empty line and, on the last line, the generated m.

    # Write random H to "input.dat"
    input_filename = "input.dat"
    with open(input_filename, 'w') as f:
        for i in range(H.shape[0]):
            row_str = ''.join(str(x) for x in H[i])
            f.write(row_str + "\n")
        f.write("\n")
        # Write random m to "input.dat"
        f.write(''.join(str(x) for x in m) + "\n")

    # Convert each row of random H to an integer.
    H_int = []
    for row in H:
        # Convert row (array of 0s and 1s) to a binary string.
        bin_str = ''.join(str(bit) for bit in row)
        # Convert the binary string to a decimal integer.
        row_int = int(bin_str, 2)
        H_int.append(row_int)

    #Convert m to an integer
    bin_str = ''.join(str(bit) for bit in m)
    m_int = int(bin_str,2)

    input_filename = "input_int.dat"
    with open(input_filename, 'w') as f:
        # Write converted rows of random H to "input.dat"
        for i in range(len(H_int)):
            f.write(str(H_int[i]) + "\n")

        f.write("\n")
        # Write converted m to "input.dat"
        f.write(str(m_int) + "\n")

    print(f"\n{'-' * 32}")

    # Calculate y and t.
    print("\nCalculating y")
    y = compute_y(H, m)
    leftText = "y[k] = "
    for y_val in y:
        print(f"{leftText}[{y_val}]")
        leftText = "       "

    print("\nDetermining Hamming weight:")
    t = int(np.sum(m))
    print(f"t = {t}")

    # Write the output to "output.dat".
    # For each row of H, write the concatenated binary digits
    #  with the corresponding y bit appended and separated with space.
    # Then write an empty line and, on the last line, the value of t.
    output_filename = "output.dat"
    with open(output_filename, 'w') as f:
        for i in range(H.shape[0]):
            row_str = ''.join(str(x) for x in H[i])
            line = row_str + " " + str(y[i])
            f.write(line + "\n")
        f.write("\n")
        f.write(str(t) + "\n")

    return True

