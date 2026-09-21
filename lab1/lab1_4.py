import numpy as np

INPUT_FILE_PATH = "data/input_4.txt"


def euclidian_norm(array):
    return np.sqrt((array.flatten() ** 2).sum())


def read_array(input_file_path):
    with open(input_file_path, mode="r") as file:
        content = np.array([line.split() for line in file], dtype=float)

    return content


def rotating_method(array, eps=0.01):
    array = array.copy()

    if not np.array_equal(array, array.T):
        raise ValueError("Matrix should be simmetric")

    n = array.shape[0]
    U = np.eye(n)
    while True:
        U_t = np.eye(n)
        a = -np.inf
        i = 0
        j = 0
        for k in range(n):
            for m in range(n):
                if np.abs(array[k, m]) > a and k != m:
                    a = np.abs(array[k, m])
                    i = k
                    j = m

        if array[i, i] == array[j, j]:
            phi = np.pi / 4
        else:
            phi = 0.5 * np.arctan((2 * array[i, j]) / (array[i, i] - array[j, j]))
        U_t[i, i] = np.cos(phi)
        U_t[j, j] = np.cos(phi)
        U_t[i, j] = -np.sin(phi)
        U_t[j, i] = np.sin(phi)
        array = U_t.T @ array @ U_t
        U = U @ U_t
        if euclidian_norm(array[~np.eye(n, dtype=bool)]) < eps:
            break

    return U, array


def main():
    array = read_array(INPUT_FILE_PATH)
    U, diag_array = rotating_method(array, eps=1e-4)
    eigenvectors = U.T
    eigenvalues = np.diag(diag_array)
    print("Eigenvectors:")
    for i in range(U.shape[0]):
        print(U[i], end=" ")

    print("\nEigenvalues:\n", eigenvalues)

    print("Validation:")
    for value, vector in zip(eigenvalues, eigenvectors):
        print(array @ vector, "\t", value * vector)
        assert np.allclose(array @ vector, value * vector, atol=1e-4)

    print("Numpy:")
    eigenvalues, eigenvectors = np.linalg.eigh(array)
    print("Eigenvectors:")
    for i in range(U.shape[0]):
        print(U[i], end=" ")

    print("\nEigenvalues:\n", eigenvalues)


if __name__ == "__main__":
    main()
