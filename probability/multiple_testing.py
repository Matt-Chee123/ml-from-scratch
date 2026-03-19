import numpy as np
from typing import Tuple
from t_test_estimation import simulate_null_t_tests

def bonferroni_correction(p_values, alpha):
    adjusted_threshold = alpha / len(p_values)

    rejected_p_vals = p_values < adjusted_threshold

    return rejected_p_vals, adjusted_threshold

def bh(p_values, q):
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    m = len(p_values)
    k = -1

    for i in range(m):
        if sorted_p[i] <= ((i + 1) / m) * q:
            k = i
    rejected = np.zeros(m, dtype=bool)
    if k >= 0:
        rejected[sorted_indices[:k+1]] = True

    return rejected

if __name__ == "__main__":
    p_values = np.array([0.001, 0.01, 0.03, 0.04, 0.05,
                          0.10, 0.12, 0.20, 0.35, 0.50])

    print("=== Multiple Testing Corrections ===")
    # demonstrate_corrections(p_values, alpha=0.05)

    print("\n=== Applied to 1,000 null t-tests ===")

    null_p = simulate_null_t_tests(n_tests=900)

    signal_p = np.random.uniform(0, 0.001, size=100)

    p_values = np.concatenate([null_p, signal_p])

    rejected_bonf, _ = bonferroni_correction(p_values, alpha=0.05)
    rejected_bh = bh(p_values, q=0.05)

    print(f"Raw rejections (no correction): {(null_p < 0.05).sum()}")
    print(f"After Bonferroni:               {rejected_bonf.sum()}")
    print(f"After BH:                       {rejected_bh.sum()}")
    print("All three should be roughly ~50, ~0-5, ~50 respectively.")

