import numpy as np

def power_iteration(A, num_iterations=100, tol=1e-6):
    b0 = np.random.rand(A.shape[1])

    for n in range(num_iterations):
        bk = np.matmul(A, b0)
        bnorm = np.linalg.norm(bk)
        bk1 = bk / bnorm

        if np.linalg.norm(bk1 - b0) < tol:
            print("Broken")
            break
        b0 = bk1

    eigenvalue = np.dot(b0.T, np.matmul(A, b0))
    return eigenvalue, b0

def deflate(A, eigenvalue, eigenvector):
    return A - eigenvalue * np.outer(eigenvector, eigenvector.T)

def top_k_eigenvectors(A, k):
    eigenvectors = []
    A_copy = A.copy()
    while len(eigenvectors) != k:
        eigenvalue, eigenvector = power_iteration(A_copy)
        eigenvectors.append(eigenvector)
        A_copy = deflate(A_copy, eigenvalue, eigenvector)
    return eigenvectors

A = np.array([[2, 1],
              [1, 2]])

eigenvalue, eigenvector = power_iteration(A)
deflated = deflate(A, eigenvalue, eigenvector)

eigenvectors = top_k_eigenvectors(A, 2)

print("Eigenvalue:", eigenvalue)
print("Eigenvector:", eigenvector)
print("Deflated:", deflated)
print("Top 10 eigenvectors:", eigenvectors)