import numpy as np
import matplotlib.pyplot as plt

def lln_demo():
    rng = np.random.default_rng(42)
    N = 10000

    bern = rng.binomial(1, 0.7, size=N)
    pois = rng.poisson(3, size=N)

    bern_running_mean = np.cumsum(bern) / np.arange(1 , N + 1)
    pois_running_mean = np.cumsum(pois) / np.arange(1 , N + 1)

    pois_running_mean_sqr = np.cumsum(pois**2) / np.arange(1, N + 1)
    pois_running_var = pois_running_mean_sqr - pois_running_mean ** 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(bern_running_mean, color='#4a90d9', linewidth=0.8)
    axes[0].axhline(0.7, color='red', linewidth=1.5, linestyle='--', label='True mean=0.7')
    axes[0].axhline(0.75, color='gray', linewidth=0.8, linestyle=':')
    axes[0].axhline(0.65, color='gray', linewidth=0.8, linestyle=':')
    axes[0].set_title('LLN — Bernoulli(p=0.7)')
    axes[0].set_xlabel('n samples')
    axes[0].set_ylabel('Running mean')
    axes[0].legend()

    axes[1].plot(pois_running_mean, color='#50c0a0', linewidth=0.8)
    axes[1].axhline(3, color='red', linewidth=1.5, linestyle='--', label='True mean=3')
    axes[1].set_title('LLN — Poisson(λ=3)')
    axes[1].set_xlabel('n samples')
    axes[1].legend()

    axes[2].plot(pois_running_var, color='#e0c050', linewidth=0.8)
    axes[2].axhline(3, color='red', linewidth=1.5, linestyle='--', label='True var=3')
    axes[2].set_title('LLN — Poisson variance (= mean = 3)')
    axes[2].set_xlabel('n samples')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

lln_demo()