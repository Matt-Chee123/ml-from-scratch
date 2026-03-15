import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from poisson_limit import binomial_pmf


def verify_bernoulli_binomial():
    rng = np.random.default_rng(42)
    p, n = 0.3, 20

    bern_samples = rng.binomial(1, p, size=100000)
    binom_samples = rng.binomial(n, p, size=100000)

    print("Bernoulli Theoretical E[X]: ", p)
    print("Bernoulli Simulated E[X]: ", np.mean(bern_samples))
    print("Bernoulli Theoretical Var: ", p * (1 - p))
    print("Bernoulli Simulated Var: ", np.var(bern_samples))

    print("Binomial Theoretical E[X]: ", n * p)
    print("Binomial Simulated E[X]: ", np.mean(binom_samples))
    print("Binomial Theoretical Var: ", n * p * (1 - p))
    print("Binomial Simulated Var: ", np.var(binom_samples))

def verify_poisson():
    rng = np.random.default_rng(42)
    lam = 5
    samples = 10000

    poisson_samples = rng.poisson(lam, samples)

    print("Poisson Theoretical E[X] and Var: ", lam)
    print("Poisson Simulated E[X]: ", np.mean(poisson_samples))
    print("Poisson Simulated Var: ", np.var(poisson_samples))

def binomial_coefficient_pmf(k, n, p):
    return binomial_pmf(k, n, p)

def plot_binomial():
    rng = np.random.default_rng(42)
    p = 0.3

    fig, ax = plt.subplots(figsize=(10,5))
    colors = ['#4a90d9', '#50c0a0', '#e0c050']

    for n, color in zip([5, 20, 100], colors):
        k_vals = np.arange(0, n + 1)
        pmf = [binomial_pmf(k, n, p) for k in k_vals]
        ax.plot(k_vals, pmf, 'o-', color=color, alpha=0.7, label=f'Binomial(n={n})')

    mu, sigma = 100 * p, np.sqrt(100 * p * (1 - p))
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
    ax.plot(x, norm.pdf(x, mu, sigma), 'k--', linewidth=2,
            label=f'Normal(μ={mu}, σ={sigma:.2f})')

    ax.set_title('Binomial → Normal as n grows (CLT preview)')
    ax.set_xlabel('k')
    ax.legend()
    plt.tight_layout()
    plt.show()

def summary_table():
    rng = np.random.default_rng(42)
    N = 100_000
    rows = [
        ("Bernoulli(0.3)",    rng.binomial(1, 0.3, N),      0.3,    0.3*0.7),
        ("Binomial(20,0.3)",  rng.binomial(20, 0.3, N),     6.0,    4.2),
        ("Poisson(5)",        rng.poisson(5, N),             5.0,    5.0),
    ]
    print(f"\n{'Distribution':<20} {'Theo mean':>10} {'Sim mean':>10} {'Theo var':>10} {'Sim var':>10}")
    print("-" * 65)
    for name, samples, t_mean, t_var in rows:
        print(f"{name:<20} {t_mean:>10.3f} {samples.mean():>10.3f} {t_var:>10.3f} {samples.var():>10.3f}")


verify_bernoulli_binomial()
verify_poisson()
plot_binomial()
summary_table()
