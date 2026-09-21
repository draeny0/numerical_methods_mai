import numpy as np

INPUT_FILE_PATH = "data/input_5.txt"


def euclidian_norm(array):
    return np.sqrt((array.flatten() ** 2).sum())


def read_array(input_file_path):
    with open(input_file_path, mode="r") as file:
        array = np.array([line.split() for line in file], dtype=float)

    return array

def converged(A, eps):
    n = A.shape[0]
    i = 0
    while i < n - 1:
        if abs(A[i+1:, i]).sum() < eps:
            i += 1
        else:
            if i + 2 >= n:
                i += 2
            else:
                if abs(A[i+2:, i+1]).sum() < eps:
                    i += 2
                else:
                    return False
    return True


def eigvalues(array, eps=1e-3):
    n, m = array.shape
    if n != m:
        raise ValueError("Matrix should be square")

    A = array.copy()
    while True:
        Q, R = qr_decomposition(A)
        A = R @ Q
        if converged(A, eps):
            break

    eigvalues = np.empty(n, dtype=np.complex128)
    i = 0
    while i < n:
        if i == n - 1:
            eigvalues[i] = A[i, i]
            i += 1
            continue

        if abs(A[i + 1, i]) < eps:
            eigvalues[i] = A[i, i]
            i += 1
            continue

        a = np.complex128(A[i, i])
        b = np.complex128(A[i, i + 1])
        c = np.complex128(A[i + 1, i])
        d = np.complex128(A[i + 1, i + 1])

        descriminant = np.sqrt((a + d) ** 2 - 4 * (a * d - b * c))
        eigvalues[i] = ((a + d) + descriminant) * 0.5
        eigvalues[i + 1] = ((a + d) - descriminant) * 0.5
        i += 2

    return eigvalues


def qr_decomposition(array):
    n, m = array.shape
    if n != m:
        raise ValueError("Matrix should be square")

    A = array.copy()
    e = np.eye(n)
    Q = np.eye(n)
    for i in range(n - 1):
        b = A[:, i]
        v = np.zeros_like(b)
        v[i:] = b[i:] + np.sign(b[i:]) * euclidian_norm(b[i:]) * e[i:, i]
        H_i = np.eye(n) - 2 * (v[:, None] @ v[None]) / (v[None] @ v[:, None])
        Q = Q @ H_i
        A = H_i @ A

    return Q, A


def main():
    array = read_array(INPUT_FILE_PATH)
    print("Matrix:\n", array)
    q, r = qr_decomposition(array)
    print("Q:\n", q)
    print("R:\n", r)
    print("Q @ R:\n", q @ r)
    print("Eigvalues:\n", eigvalues(array, eps=1e-2))


if __name__ == "__main__":
    main()

# №6
# 8 -1 -3
# -5 9 -8
# 4 -5 7


# №7
# 9 0 2
# -6 4 4
# -2 -7 5