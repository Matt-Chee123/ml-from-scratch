import numpy as np

def svd_from_scratch(A, k=3):
    V = []
    s = []
    for _ in range(k):
        bk = np.random.rand(A.shape[1])
        for _ in range(200):
            bk = A.T @ (A @ bk)

            for v in V:
                bk -= np.dot(bk, v) * v

            bk = bk / np.linalg.norm(bk)

        sigma = np.linalg.norm(A @ bk)

        V.append(bk)
        s.append(sigma)

    V = np.array(V).T
    U = A @ V / s

    return U, s, V.T

np.random.seed(0)
A = np.random.randn(50, 20)

k = 5
U, s, Vt = svd_from_scratch(A, k=k)
s_np = np.linalg.svd(A, full_matrices=False)[1][:k]

print(f"Yours:  {np.round(s, 4)}")
print(f"Numpy:  {np.round(s_np, 4)}")
print(f"Max diff: {np.max(np.abs(s - s_np)):.2e}")
assert np.max(np.abs(s - s_np)) < 1e-4, "FAILED: singular values don't match"
print("Singular values PASSED ✓")
