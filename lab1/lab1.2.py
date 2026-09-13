import numpy as np


INPUT_FILE_NAME = 'lab1.2_input.txt'


def read_array(file_name):
    with open(file_name, mode='r') as file:
        array = np.array([line.split() for line in file], dtype=float)
    return array


def tridiagonal_matrix_algorithm(matrix):
    m, n = matrix.shape

    if (n != 4 or
            matrix[0, 0] != 0 or
            matrix[m - 1, n - 2] != 0):
        message = '''
Please enter matrix kind of:
    0 * * *
    * * * *
    - - - -
    * * 0 *
        '''
        raise ValueError(message)
    
    P = np.zeros(m)
    Q = np.zeros(m)

    P[0] = -matrix[0, 2] / matrix[0, 1]
    Q[0] = matrix[0, 3] / matrix[0, 1]
    for i in range(1, m):
        

        P[i] = -matrix[i, 2] / (matrix[i, 1] + matrix[i, 0] * P[i - 1])
        Q[i] = (matrix[i, 3] - matrix[i, 0] * Q[i - 1]) / (matrix[i, 1] + matrix[i, 0] * P[i - 1])

    x = np.zeros(m)
    x[m - 1] = Q[m - 1]
    for i in range(n - 2, -1, -1):
        x[i] = x[i + 1] * P[i] + Q[i]

    return x, P, Q


def main():
    array = read_array(INPUT_FILE_NAME)
    try:
        x, P, Q = tridiagonal_matrix_algorithm(array)
        print('P:', P)
        print('Q:', Q)
        print('x:', x)

    except ValueError as error:
        print(error)
    

if __name__ == "__main__":
    main()