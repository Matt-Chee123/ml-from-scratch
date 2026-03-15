import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def sum_of_uniforms(n, n_samples, rng):
    return rng.uniform(0, 1, (n_samples, n)).sum(axis=1)

def sum_of_bernoulli(n, n_samples, p, rng):
    return rng.binomial(1, p, (n_samples, n)).sum(axis=1)

def plot_clt_uniform():
    rng = np.random.default_rng(42)
    n_values = [1, 2, 5, 10, 30]
    n_samples = 50000
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    x = np.linspace(-1, 35, 500)

    for ax, n in zip(axes, n_values):
        samples = sum_of_uniforms(n, n_samples, rng)
        mu, var = n / 2, n / 12
        ax.hist(samples, bins=60, density=True,
                color='#4a90d9', alpha=0.6, label='Empirical')
        x_plot = np.linspace(samples.min(), samples.max(), 300)
        ax.plot(x_plot, stats.norm.pdf(x_plot, mu, np.sqrt(var)),
                'r-', linewidth=2, label=f'N({mu:.1f},{var:.2f})')
        ax.set_title(f'n={n}')
        ax.legend(fontsize=7)

    plt.suptitle('CLT: Sum of n Uniform(0,1) → Normal')
    plt.tight_layout()
    plt.show()

def plot_clt_bernoulli():
    rng = np.random.default_rng(42)
    p = 0.2
    n_values = [1, 2, 5, 10, 30]
    n_samples = 50_000

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, n in zip(axes, n_values):
        samples = sum_of_bernoulli(n, n_samples, p, rng)
        mu, var = n * p, n * p * (1 - p)

        ax.hist(samples, bins=min(n+1, 30), density=True,
                color='#50c0a0', alpha=0.6, label='Empirical')
        x_plot = np.linspace(samples.min(), samples.max(), 300)
        ax.plot(x_plot, stats.norm.pdf(x_plot, mu, np.sqrt(var)),
                'r-', linewidth=2)
        ax.set_title(f'n={n}')
    plt.suptitle('CLT: Sum of n Bernoulli(p=0.2) → Normal (slower convergence)')
    plt.tight_layout()
    plt.show()

plot_clt_uniform()
plot_clt_bernoulli()