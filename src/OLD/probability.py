import sys
import numpy as np

from generate import generate_random_H, generate_random_m, compute_y

# Define size of random matrix H (n columns and k rows)
n, k = 2000, 1000

# Generate random matrix H
H = generate_random_H(k, n)

for t in range(2,21):
    print(f"\nt={(2 - len(str(t))) * ' '}{str(t)}: ", end="")
    sys.stdout.flush()

    sol_cnt = 0
    for i in range(0, 100000):
        isSolution = 0

        # Generate random m with weight t
        m = generate_random_m(n, t)

        # Calculate y list
        y = compute_y(H, m)

        # Calculate Phi
        y_arr = np.array(y)
        Phi = []
        for H_col in range(H.shape[1]):
            dot_val = np.dot(y_arr, H[:, H_col])
            Phi.append((H_col, dot_val))

        # Calculate the complementary matrix of H
        H_c = 1 - H

        # Calculate y_c
        y_c = [t - y_val for y_val in y]

        # Calculate Phi_c
        yc_arr = np.array(y_c)
        Phi_c = []
        for H_col in range(H_c.shape[1]):
            dot_val = np.dot(yc_arr, H_c[:, H_col])
            Phi_c.append((H_col, dot_val))

        # Calculate Score vector: for each index, sum the corresponding dot products from Phi and Phi_c.
        Score = []
        for cnt in range(len(Phi)):
            # Both Phi and Phi_c are lists of tuples (index, value)
            score_val = Phi[cnt][1] + Phi_c[cnt][1]
            Score.append((Phi[cnt][0], score_val))

        # Sort Score vector descending into Score_sorted.
        # Then, mark the first t elements by adding an "*" to each index.
        Score_sorted = sorted(Score, key=lambda x: x[1], reverse=True)

        # Generate a new "m" vector based on Score_sorted:
        # For each column, set 1 in positions indicated by the indices of the first t tuples in Score_sorted,
        # and 0 in all other positions.
        new_m = [0] * H.shape[1]
        for idx, (cnt, val) in enumerate(Score_sorted):
            if idx < t:
                new_m[cnt] = 1

        # Compute "new_y" from "H" and "new_m"
        new_y = compute_y(H, new_m)

        if y == new_y:
            sol_cnt += 1
            isSolution = 1

        print(f"\rt={(2 - len(str(t))) * ' '}{str(t)}: {sol_cnt}/{i + 1} solutions", end="")

        sys.stdout.flush()


