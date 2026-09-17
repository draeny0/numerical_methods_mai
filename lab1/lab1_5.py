import numpy as np


INPUT_FILE_PATH = "data/input_5.txt"


def euclidian_norm(array):
    return np.sqrt((array.flatten() ** 2).sum())


def read_array(input_file_path):
    with open(input_file_path, mode="r") as file:
        array = np.array([line.split() for line in file], dtype=float)

    return array


def eigvalues(array, eps=1e-3):
    n, m = array.shape
    if n != m:
        raise ValueError("Matrix should be square")

    A = array.copy()
    n_iter = 0
    while n_iter != 10:
        n_iter += 1
        Q, R = qr_decomposition(A)
        A = R @ Q
        print(A, end="\n\n")
        dim_indices = np.indices((n, n), sparse=True)
        if euclidian_norm(A[dim_indices[0] > dim_indices[1]]) < eps:
            break

    eigvalues = np.empty(n, dtype=np.complex128)
    i = 0
    while i < n:
        if euclidian_norm(A[i + 1 :, i]) < eps:
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
        v = b + np.sign(b) * euclidian_norm(b[i:]) * e[:, i]
        v[:i] = 0
        H_i = np.eye(n) - 2 * (v[:, None] @ v[None]) / (v[None] @ v[:, None]).item()
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
    print("Eigvalues:\n", eigvalues(array))


if __name__ == "__main__":
    main()
