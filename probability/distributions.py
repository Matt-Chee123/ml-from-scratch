import math

import numpy as np
import matplotlib.pyplot as plt

#Benroulli function
def bernoulli_pmf(p):
    assert 0 <= p <= 1
    return {0: 1-p, 1: p}

print("Bernoulli Dist: ", bernoulli_pmf(0.6))

#Binomial function
def binomial_pmf(n, p, k):
    if k < 0 or k > n:
        return 0.0

    combinations = math.comb(n, k)
    return combinations * p ** k * (1 - p) ** (n - k)

def binomial_pmf_array(n, p):
    list = []
    for k in range(n + 1):
        list.append(binomial_pmf(n, p, k))

    return np.array(list)

print("Binomial Probability n = 10, p = 0.3, k = 4: ", binomial_pmf(10, 0.3, 4))
print("Binomial List n = 10, p = 0.1: ", binomial_pmf_array(5, 0.4))


#poisson distribution
def poisson_pmf(lam, k):
    if k < 0:
        return 0.0

    return (lam ** k) / math.factorial(k) * np.exp(-lam)

def poisson_pmf_array(lam, max_k):
    list = []
    for k in range(max_k + 1):
        list.append(poisson_pmf(lam, k))

    return np.array(list)

print(f"\nPoisson(λ=5, k=5): {poisson_pmf(5, 5):.6f}")
print("\nPoisson Distribution(λ=5, max_k=3): ", poisson_pmf_array(5, 3))

#convergence plot
def plot_convergence():
    params = [
        (10,   0.5,   "n=10, p=0.5  (poor approximation)"),
        (50,   0.1,   "n=50, p=0.1  (reasonable)"),
        (1000, 0.002, "n=1000, p=0.002  (near-perfect)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Binomial(n,p) → Poisson(λ=np) as n→∞, p→0", fontsize=14)

    for ax, (n, p, title) in zip(axes, params):
        lam = n * p
        k_vals = np.arange(0, n + 1)

        binom_probs = binomial_pmf_array(n, p)
        cutoff = np.where(binom_probs > 1e-6)[0]
        if len(cutoff) == 0:
            continue
        k_plot = np.arange(cutoff[0], cutoff[-1] + 1)

        binom_plot = binom_probs[k_plot]
        poisson_plot = poisson_pmf_array(lam, int(k_plot[-1]))[k_plot]

        ax.bar(k_plot, binom_plot, color="steelblue", alpha=0.6,
               label=f"Binomial(n={n}, p={p})")
        ax.plot(k_plot, poisson_plot, "ro-", markersize=4,
                label=f"Poisson(λ={lam:.1f})")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("k")
        ax.set_ylabel("P(X=k)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

plot_convergence()

#poisson mean variance validation
def poisson_mean_variance_check(lam=5, n_samples=100_000):
    rng = np.random.default_rng(seed=42)
    samples = rng.poisson(lam, n_samples)
    sample_mean = samples.mean()
    sample_var  = samples.var()
    print(f"λ = {lam}")
    print(f"Sample mean:     {sample_mean:.4f}  (expected ≈ {lam})")
    print(f"Sample variance: {sample_var:.4f}  (expected ≈ {lam})")
    print(f"Ratio var/mean:  {sample_var / sample_mean:.4f}  (expected ≈ 1.0)")

poisson_mean_variance_check()