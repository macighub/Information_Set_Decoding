import sys
import numpy as np
import multiprocessing as mp
from cls_generate import generate_random_H, generate_random_m, compute_y

# Define size of random matrix H (n columns and k rows)
n, k = 2000, 1000
num_processes = 10  # Number of sub-processes per t

# Generate random matrix H (Shared by all processes)
H = generate_random_H(k, n)

def display_vector(vector):
    """Displays the current state of the shared vector in real-time."""
    sys.stdout.write("\r" + "\n".join(f"t={(2 - len(str(t + 1))) * ' '}{str(t + 1)}: {vector[t]}" for t in range(20)))
    sys.stdout.write("\033[2F")
    sys.stdout.flush()

def process_chunk(t, start_i, end_i, vector, lock):
    """Processes a chunk of iterations for a given value of t and updates shared vector."""
    local_iterations = 0
    local_solutions = 0
    add_iterations = 0
    add_solutions = 0
    display_count = 0

    for i in range(start_i, end_i + 1):
        display_count += 1
        local_iterations += 1  # Track iterations
        add_iterations += 1
        #with lock:
        #    vector[t-1] = (vector[t-1][0] + 1, vector[t-1][1])

        # Generate random m with weight t
        m = generate_random_m(n, t)

        # Calculate y list
        y = compute_y(H, m)

        # Calculate Phi
        y_arr = np.array(y)
        Phi = [(H_col, np.dot(y_arr, H[:, H_col])) for H_col in range(H.shape[1])]

        # Calculate the complementary matrix of H
        H_c = 1 - H

        # Calculate y_c
        y_c = [t - y_val for y_val in y]

        # Calculate Phi_c
        yc_arr = np.array(y_c)
        Phi_c = [(H_col, np.dot(yc_arr, H_c[:, H_col])) for H_col in range(H_c.shape[1])]

        # Calculate Score vector
        Score = [(Phi[cnt][0], Phi[cnt][1] + Phi_c[cnt][1]) for cnt in range(len(Phi))]

        # Sort Score vector descending
        Score_sorted = sorted(Score, key=lambda x: x[1], reverse=True)

        # Generate new "m" vector based on Score_sorted
        new_m = [0] * H.shape[1]
        for idx, (cnt, val) in enumerate(Score_sorted[:t]):
            new_m[cnt] = 1

        # Compute "new_y" from "H" and "new_m"
        new_y = compute_y(H, new_m)

        if y == new_y:
            local_solutions += 1  # Track solutions
            add_solutions += 1

        # Display the updated vector in real-time
        if display_count == 100:
            with lock:
                vector[t-1] = (vector[t-1][0] + add_iterations, vector[t-1][1] + add_solutions)
            add_iterations = 0
            add_solutions = 0
            display_count = 0
            display_vector(vector)

    ## Safely update shared vector
    #with lock:
    #    vector[t-1] = (vector[t-1][0] + local_iterations, vector[t-1][1] + local_solutions)


def process_t(t, vector, lock):
    """Runs 100 processes in parallel for a given value of t."""
    total_iterations = 1000000
    chunk_size = total_iterations // num_processes

    processes = []
    for i in range(num_processes):
        start_i = i * chunk_size + 1
        end_i = (i + 1) * chunk_size
        p = mp.Process(target=process_chunk, args=(t, start_i, end_i, vector, lock))
        p.start()
        processes.append(p)

    print(f"Processes for t={(2 - len(str(t))) * ' '}{str(t)} started")


    for p in processes:
        p.join()

    print(f"\nt={(2 - len(str(t))) * ' '}{str(t)}: {vector[t-1][1]}/{vector[t-1][0]} solutions", end="")
    sys.stdout.flush()

if __name__ == "__main__":
    mp.set_start_method("spawn")  # Fix for macOS/Linux

    # Create Manager inside __main__
    with mp.Manager() as manager:
        vector = manager.list([(0, 0)] * 20)  # Initialize shared vector
        lock = manager.Lock()  # Ensure atomic updates

        processes = []
        for t in range(1,21):
            p = mp.Process(target=process_t, args=(t, vector, lock))
            print(f"Starting processes for t={(2 - len(str(t))) * ' '}{str(t)}")
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print("\nAll computations completed.")