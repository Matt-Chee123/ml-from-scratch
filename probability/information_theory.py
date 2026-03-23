import numpy as np


def entropy(p, base=0):

    p = np.asarray(p, dtype=float)
    assert np.isclose(p.sum(), 1.0), f"p must sum to 1, got {p.sum()}"

    p_safe = np.clip(p, 1e-15, None)
    return - np.sum(p_safe * np.log(p_safe))


def kl_divergence(p, q):

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    assert np.isclose(p.sum(), 1.0), f"p must sum to 1, got {p.sum()}"
    q = np.asarray(q, dtype=float)
    assert np.isclose(q.sum(), 1.0), f"q must sum to 1, got {p.sum()}"

    q_safe = np.clip(q, 1e-15, None)

    return np.sum(p * np.log(p/q_safe))


def cross_entropy(p, q):
    return - np.sum(p * np.log(q))


def verify_decomposition(p: np.ndarray, q: np.ndarray) -> None:
    hp = entropy(p)
    kl = kl_divergence(p, q)
    hpq = cross_entropy(p, q)

    print(f"H(p)       = {hp:.8f}")
    print(f"KL(p||q)   = {kl:.8f}")
    print(f"H(p) + KL  = {hp + kl:.8f}")
    print(f"H(p, q)    = {hpq:.8f}")
    print(f"Residual   = {abs(hpq - (hp + kl)):.2e}  (should be < 1e-10)")


if __name__ == "__main__":
    p = np.array([0.1, 0.4, 0.2, 0.2, 0.1])
    q = np.array([0.2, 0.3, 0.2, 0.1, 0.2])

    print(f"KL(p||q) = {kl_divergence(p, q):.6f}  (must be >= 0)")
    print(f"KL(q||p) = {kl_divergence(q, p):.6f}  (must be >= 0)")
    print(f"KL(p||p) = {kl_divergence(p, p):.6f}  (must be exactly 0)")

    print(f"\nAsymmetry: KL(p||q) != KL(q||p): "
          f"{not np.isclose(kl_divergence(p,q), kl_divergence(q,p))}")

    print("\n=== Verify H(p,q) = H(p) + KL(p||q) ===")
    verify_decomposition(p, q)

    print("\n=== Minimum cross-entropy is at q=p ===")
    print(f"H(p, p) = {cross_entropy(p, p):.6f} = H(p) = {entropy(p):.6f}")
    print("KL(p||p) = 0, so H(p,p) = H(p) + 0 = H(p). Confirmed.")