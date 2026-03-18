
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def simulate_null_t_tests(n_tests=1000, sample_size=50, alpha=0.05,seed=42):
    rng = np.random.default_rng(seed)
    p_values = np.zeros(n_tests)

    for i in range(n_tests):
        group_a = rng.normal(0, 1, size=sample_size)
        group_b = rng.normal(0, 1, size=sample_size)
        _, p_values[i] = stats.ttest_ind(group_a, group_b)

    print("False positive rate: ", np.mean(p_values < alpha))

    plt.hist(p_values, bins=50, density=True)
    plt.axhline(1, color='red', linestyle='--', label='Uniform(0,1)')
    plt.title("P-values under Null Hypothesis")
    plt.legend()
    plt.show()

    return p_values


def report_false_positive_rate(p_values, alpha=0.05):
    n_significant = len([x for x in p_values if x < alpha])
    n_total = len(p_values)

    print(f"Significant results (p < {alpha}): {n_significant}/{n_total} "
          f"= {100*n_significant/n_total:.1f}%")
    print(f"Expected under H₀: {100*alpha:.1f}%")

    print("Empirical number: ", n_significant, " Expected Number: ", n_total * alpha)


def clt_simulation(n_vars_list=[1, 5, 30, 100], n_repetitions=10000):
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, len(n_vars_list), figsize=(16, 4))

    for ax, n in zip(axes, n_vars_list):
        samples = rng.uniform(0, 1, size=(n_repetitions,n)).sum(axis=1)
        z = (samples - 0.5 * n) / np.sqrt(n * 1/12)

        ax.hist(z, bins=50, density=True, alpha=0.7, label=f'n={n}')

        x = np.linspace(-4, 4, 200)
        ax.plot(x, stats.norm.pdf(x), 'r-', lw=2, label='N(0,1)')
        ax.set_title(f'Sum of {n} Uniform(0,1)')
        ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== Null t-test simulation ===")
    p_values = simulate_null_t_tests()
    report_false_positive_rate(p_values)

    print("\n=== CLT simulation ===")
    clt_simulation()

    print("\nKey insight: even with NO true effect, ~5% of tests are significant.")
    print("Run 20 tests and you have a ~64% chance of at least one false positive.")