from operator import truediv

import numpy as np
import re

from OLD.generate import compute_y


def read_parity_matrix(filename):
    """
    Reads a parity-check matrix H and a list y from a file.

    Processing of each non-empty line in the file:
      1. The leading part of the line is scanned for characters '0' and '1'
         (stopping as soon as a different character is encountered). This
         sequence becomes one row candidate for H.
      2. The value of n (the number of columns for H) is determined as the
         maximum length of these leading sequences over all lines.
      3. Each candidate row is padded with zeros on the right if its length is
         less than n.
      4. The y value is taken as the last contiguous group of digits
         (which may include digits other than 0 and 1) found at the end of the
         line. That is, we extract the number that appears after the last
         non-numeric character.

    Returns:
      - H: A NumPy array of shape (number_of_lines, n) representing the parity-check matrix.
      - y: A list of integer values (one per row).
      - n: The number of columns in H.
    """

    candidate_rows = []  # To store the leading 0/1 sequences from each line.
    y_list = []  # To store the y integer value from each line.
    t_value = 0 # To store the value of t

    with open(filename, 'r') as f:
        last_line = ''
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace.
            if not line:
                continue  # Skip empty lines.
            else:
                last_line = line

            # Extract the sequence of '0' and '1'
            binary_digits = []
            for ch in line:
                if ch in ('0', '1'):
                    binary_digits.append(ch)
                else:
                    # Stop at the first character that is not '0' or '1'
                    break
            # Append the leading binary digit sequence
            if len(binary_digits) > 0:
                candidate_rows.append(binary_digits)

                # Extract the y value: the last contiguous block of digit characters at the end of the line.
                match = re.search(r'(\d+)\s*$', line)
                if match:
                    y_value = int(match.group(1))
                    y_list.append(y_value)
                else:
                    # If no trailing number is found we assign 0.
                    y_list.append(0)

        # Extract the value of t: the last contiguous block of digit characters at the end of the last line.
        match = re.search(r'(\d+)\s*$', last_line)
        if match:
            t_value = int(match.group(1))

    if not candidate_rows:
        raise ValueError("No valid lines found in the file.")

    # Determine n: For each row, count the number of leading digits and take the maximum.
    n = max(len(row) for row in candidate_rows)

    # Build the parity-check matrix H
    H_rows = []
    for row in candidate_rows:
        # Convert the list of digit characters to integer values.
        int_row = [int(x) for x in row]
        # If the row is shorter than n, pad it with zeros.
        if len(int_row) < n:
            int_row.extend([0] * (n - len(int_row)))
        H_rows.append(int_row[:n])

    H = np.array(H_rows, dtype=int)
    return H, y_list, t_value

def isd(filename = 'output.dat'):
    print(f"\n{'=' * 32}")
    print("\nLoading Parity matrix (H), y list (y) and Hamming weight (t):")

    try:
        H, y, t = read_parity_matrix(filename)

        leftText = "\nH[k,n] = "
        for H_row in H:
            print(f"{leftText}{H_row}")
            leftText = "         "

        leftText = "\ny[k] = "
        for y_val in y:
            print(f"{leftText}[{y_val}]")
            leftText = "       "

        print(f"\nt = {t}")

        print(f"\n{'-' * 32}")

        # Calculate Phi: For each column of H, compute dot product with y.
        print("\nCalculating Phi vector [(index: dot product)]:")
        # Convert y into a numpy array for vectorized dot product.
        y_arr = np.array(y)
        Phi = []
        for i in range(H.shape[1]):
            dot_val = np.dot(y_arr, H[:, i])
            Phi.append((i, dot_val))
        leftText = "Phi[n] = ["
        phiText = ''
        for phi in Phi:
            if not(phiText == ''):
                phiText += ', '
            phiText += f"({phi[0]}: {phi[1]})"
        phiText = f"{leftText}{phiText}]"
        print(phiText)

        # Calculate H_c: the complementary matrix of H.
        # H_c is the element-wise complement of H: 1 becomes 0 and 0 becomes 1.
        H_c = 1 - H
        print("\nCalculating complementary matrix H_c:")
        leftText = "H_c[k,n] = "
        for row in H_c:
            print(f"{leftText}{row}")
            leftText = "           "

        # Calculate y_c: for each element in y, compute t minus that element.
        print("\nCalculating y_c list (t minus each element of y):")
        y_c = [t - y_val for y_val in y]
        leftText = "y_c[k] = "
        for yc_val in y_c:
            print(f"{leftText}[{yc_val}]")
            leftText = "         "

        # Calculate Phi_c: For each column of H_c, compute dot product with y_c.
        print("\nCalculating Phi_c vector [(index: dot product)]:")
        # Convert y_c into a numpy array for vectorized dot product.
        yc_arr = np.array(y_c)
        Phi_c = []
        for i in range(H_c.shape[1]):
            dot_val = np.dot(yc_arr, H_c[:, i])
            Phi_c.append((i, dot_val))
        leftText = "Phi_c[n] = ["
        phicText = ''
        for phi_c in Phi_c:
            if not(phicText == ''):
                phicText += ', '
            phicText += f"({phi_c[0]}: {phi_c[1]})"
        phicText = f"{leftText}{phicText}]"
        print(phicText)

        # Calculate Score vector: for each index, sum the corresponding dot products from Phi and Phi_c.
        print("\nCalculating Score vector [(index: Score)]:")
        Score = []
        for i in range(len(Phi)):
            # Both Phi and Phi_c are lists of tuples (index, value)
            score_val = Phi[i][1] + Phi_c[i][1]
            Score.append((Phi[i][0], score_val))
        leftText = "Score[n] = ["
        scoreText = ''
        for score in Score:
            if scoreText != '':
                scoreText += ', '
            scoreText += f"({score[0]}: {score[1]})"
        scoreText = f"{leftText}{scoreText}]"
        print(scoreText)

        # Sort Score vector descending into Score_sorted.
        # Then, mark the first t elements by adding an "*" to each index.
        print("\nSorting Score vector descending into Score_sorted:")
        Score_sorted = sorted(Score, key=lambda x: x[1], reverse=True)
        leftText = "Score_sorted[n] = ["
        scoreSortedText = ''
        for idx, (i, val) in enumerate(Score_sorted):
            # Mark first t elements with an asterisk.
            mark = "*" if idx < t else ""
            if scoreSortedText:
                scoreSortedText += ', '
            scoreSortedText += f"({mark}{i}: {val})"
        scoreSortedText = f"{leftText}{scoreSortedText}]"
        print(scoreSortedText)

        i, j = 0, 0
        b = True
        while b:
            b = False
            if (Score_sorted[t + (i + 1)][1] == Score_sorted[t][1]):
                i = i + 1
                b = True
            if (Score_sorted[t - (j + 1)][1] == Score_sorted[t][1]):
                j = j + 1
                b = True



        # Generate a new "m" vector based on Score_sorted:
        # For each column, set 1 in positions indicated by the indices of the first t tuples in Score_sorted,
        # and 0 in all other positions.
        new_m = [0] * H.shape[1]
        for idx, (i, val) in enumerate(Score_sorted):
            if idx < t:
                new_m[i] = 1
        print("\nGenerated new m vector based on top t indices from Score_sorted:")
        print(f"m_new[n] = {np.array(new_m)}")

        # Compute "new_y" from "H" and "new_m"
        new_y = compute_y(H, new_m)
        leftText1 = "\nnew_y[k] = "
        leftText2 = "y[k] = "
        for cnt in range(len(new_y)):
            print(f"{leftText1}[{new_y[cnt]}] {leftText2}[{y[cnt]}]")
            leftText1 = "           "
            leftText2 = "       "

        # Compare "y" and "new_y"
        if y == new_y:
            print("new_y = y => 'new_m' is a solution")
        else:
            print(f"new_y <> y => 'new_m' is \033[1mNOT\033[0m a solution" )

    except Exception as e:
        print("An error occurred:", e)
