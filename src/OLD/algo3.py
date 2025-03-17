from generate import generate_random_H, generate_random_m, compute_y
import numpy as np

n, k, t = 20, 10, 6

H = generate_random_H(k,n)

m = generate_random_m(n, t)

y = compute_y(H, m)

y_arr = np.array(y)
Phi = []
for H_col in range(H.shape[1]):
    dot_val = np.dot(y_arr, H[:, H_col])
    Phi.append((H_col, dot_val))

H_c = 1 - H

y_c = [t - y_val for y_val in y]

yc_arr = np.array(y_c)
Phi_c = []
for H_col in range(H_c.shape[1]):
    dot_val = np.dot(yc_arr, H_c[:, H_col])
    Phi_c.append((H_col, dot_val))

Score = []
for cnt in range(len(Phi)):
    # Both Phi and Phi_c are lists of tuples (index, value)
    score_val = Phi[cnt][1] + Phi_c[cnt][1]
    Score.append((Phi[cnt][0], score_val))

Score_sorted = sorted(Score, key=lambda x: x[1], reverse=True)
Score_sorted1 = np.argsort(Score)[:-1]



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

new_m = [0] * n
for idx, (cnt, val) in enumerate(Score_sorted):
    if idx < t:
        new_m[cnt] = 1


Score_index = [0] * n
for idx, (cnt, val) in enumerate(Score_sorted):
    Score_index[idx] = cnt

H1 = H[:,Score_index]

print(f"\n{Score_index}")

print(f"\n{H}")

print(f"\n{H1}")
