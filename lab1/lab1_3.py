import numpy as np
from numpy.typing import NDArray
from lab1_1 import determinant
from lab1_5 import eigvalues

INPUT_FILE_NAME = "data/input_3.txt"


def matrix_norm_c(matrix):
    return np.sum(np.abs(matrix), axis=-1).max()


def read_sle(file_path):
    with open(file_path, mode="r") as file:
        array = np.array([line.split() for line in file], dtype=float)

    return array[:, :-1].copy(), array[:, -1].copy()


def seidel(A, b, eps):
    m, n = A.shape
    if m != n:
        raise ValueError("Matrix should be square")
    if determinant(A) == 0:
        raise ValueError("Matrix is singular")

    alpha_matrix = A.copy()

    x = b.copy()
    diag = alpha_matrix[np.eye(m, dtype=bool)]
    beta = b / diag
    alpha_matrix[np.eye(m, dtype=bool)] = 0
    for i in range(m):
        alpha_matrix[i] /= -diag[i]

    alpha_norm_c = matrix_norm_c(alpha_matrix)
    if alpha_norm_c >= 1:
        raise ValueError("The method diverges for |A| >= 1")

    C = alpha_matrix.copy()
    indices = np.indices((n, n))
    C[indices[0] >= indices[1]] = 0
    c_norm = matrix_norm_c(C)
    t = 0
    x_new = x.copy()
    while True:
        t += 1
        for i in range(m):
            x_new[i] = (
                alpha_matrix[i, :i] @ x_new[:i]
                + alpha_matrix[i, i:] @ x_new[i:]
                + beta[i]
            )
        if (c_norm / (1 - alpha_norm_c)) * euclide_norm(x - x_new) < eps:
            break
        x = x_new.copy()

    return x, t


def simple_iterations(A, b, eps):
    m, n = A.shape
    if m != n:
        raise ValueError("Matrix should be square")

    alpha_matrix = A.copy()

    x = b.copy()
    diag = alpha_matrix[np.eye(m, dtype=bool)]
    beta = b / diag
    alpha_matrix[np.eye(m, dtype=bool)] = 0
    for i in range(m):
        alpha_matrix[i] /= -diag[i]

    alpha_norm_c = matrix_norm_c(alpha_matrix)
    if alpha_norm_c > 1 - 1e-13:
        raise ValueError("The method diverges for |A| >= 1")
    t = 0
    while True:
        x_new = alpha_matrix @ x + beta
        if (alpha_norm_c / (1 - alpha_norm_c)) * euclide_norm(x - x_new) < eps:
            break
        x = x_new
        t += 1

    return x_new, t


def euclide_norm(array: NDArray):
    s = 0
    for i in array.flatten():
        s += i**2
    return np.sqrt(s)


def main():
    print("\tSimple iterations results")
    A, b = read_sle(INPUT_FILE_NAME)
    x, t = simple_iterations(A, b, eps=1e-2)
    print(x, f"n_iter: {t}")

    print("\tSeidel results")
    x, t = seidel(A, b, eps=1e-2)
    print(x, f"n_iter: {t}")

    print("\tNumpy solution")
    print(np.linalg.solve(A, b))


if __name__ == "__main__":
    main()
