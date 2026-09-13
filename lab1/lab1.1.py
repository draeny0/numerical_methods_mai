import numpy as np
from numpy.typing import NDArray


INPUT_FILE_NAME = 'lab1.1_input.txt'


def read_array(file_name):
    with open(file_name, mode='r') as file:
        array = np.array([line.split() for line in file], dtype=float)
    return array


def inverse(array: NDArray, eps=1e-15):
    L, U, P = lu_decompose(array)
    if np.abs(np.prod(U, where=np.eye(U.shape[0], dtype=bool))) < eps:
        raise ValueError('The matrix is singular')
    
    m = array.shape[0]

    E = np.eye(m)
    res = np.empty_like(array)
    for i in range(m):
        x = np.empty(m)
        b = E[i]
        b = P @ b
        z = np.empty(m)
        z[0] = b[0]
        for j in range(1, m):
            z[j] = b[j] - z[:j] @ L[j][:j]

        x = np.empty(m)
        x[m - 1] = z[m - 1] / U[m - 1][m - 1]
        for j in range(m - 1, -1, -1):
            x[j] = (z[j] - x[j + 1:] @ U[j][j + 1:]) / U[j, j]
        res[:, i] = x.copy()

    return res


def determinant(array: NDArray, eps=1e-17):
    _, U, P = lu_decompose(array)
    m = U.shape[0]

    perm_count = 0
    for i in range(m):
        if P[i][i] == 1:
            continue
        j = np.argmax(P[:, i])
        P[i], P[j] = P[j].copy(), P[i].copy()
        perm_count += 1

    product = np.prod(U[np.eye(m, dtype=bool)])
    if abs(product) < eps:
        return 0

    return (-1) ** perm_count * np.prod(U[np.eye(m, dtype=bool)])


def read_sle(file_name):
    array = read_array(file_name)
    A = array[:, :-1]
    b = array[:, -1]         
    return A, b


def solve_sle_using_lu(A: NDArray, b=None):
    m, n = A.shape
    if not m == n:
        raise ValueError('Matrix A is not square')

    if b is None:
        b = np.zeros(m, 1)

    L, U, P = lu_decompose(A)
    b = P @ b
    z = np.empty(m)
    z[0] = b[0]
    for i in range(1, m):
        z[i] = b[i] - z[:i] @ L[i][:i]

    x = np.empty(m)
    x[m - 1] = z[m - 1] / U[m - 1][m - 1]
    for i in range(m - 1, -1, -1):
        x[i] = (z[i] - x[i + 1:] @ U[i][i + 1:]) / U[i, i]

    return x


def lu_decompose(array: NDArray, permut=True, eps=1e-13):
    m, n = array.shape
    if not m == n:
        raise ValueError('Matrix is not square')

    P = np.eye(m)
    L = np.eye(m)
    U = array.copy()
    for i in range(m):
        if permut:
            pivote = i + np.argmax(np.abs(U[i:, i]))
            if abs(U[pivote, i]) < eps:
                continue
        
            P[i], P[pivote] = P[pivote].copy(), P[i].copy()
            U[i], U[pivote] = U[pivote].copy(), U[i].copy()
            L[i, :i], L[pivote, :i] = L[pivote, :i].copy(), L[i, :i].copy()

        for k in range(i + 1, m):
            c = U[k, i] / U[i, i]
            L[k, i] = c
            U[k] -= U[i] * c

    return L, U, P


def main():

    menu = '''
    1. Calc LU decomposition
    2. Solve SLE
    3. Calc determinant
    4. Inverse matrix
    5. Exit
'''

    while True:
        print(menu)
        choose = input()
        try:
            match choose:
                case '1':
                    array = read_array(INPUT_FILE_NAME)
                    L, U, P = lu_decompose(array)
                    print(f'L:\n{L}\n\nU:\n{U}\n\nP:\n{P}\n\n')
                    print(f'P@L@U:\n{P @ L @ U}\n\n')
                case '2':
                    A, b = read_sle(INPUT_FILE_NAME)
                    solution = solve_sle_using_lu(A, b)
                    print(f'Solution:\n{solution}\n\nAx:\n{array @ solution}\n\n')
                    assert np.linalg.norm(solution - np.linalg.solve(A, b)) < 1e-13
                case '3':
                    array = read_array(INPUT_FILE_NAME)
                    det = determinant(array)
                    print(det)
                    assert np.linalg.norm(det - np.linalg.det(array)) < 1e-10
                case '4':
                    array = read_array(INPUT_FILE_NAME)
                    inv = inverse(array)
                    print(f'A^-1\n{inv}\n\nA @ A^-1:\n{array @ inv}')
                    assert np.linalg.norm(inv - np.linalg.inv(array)) < 1e-10
                case '5':
                    break
                case _:
                    print('Please choose correct option')
        except ValueError as error:
            print(error)
                

if __name__ == "__main__":
    main()
