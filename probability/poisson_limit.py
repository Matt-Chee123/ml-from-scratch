import numpy as np
import matplotlib.pyplot as plt


#nCk
def binomial_coefficient_log(k, n):
    coefficient = np.sum(np.log(np.arange(1, n + 1))) - np.sum(np.log(np.arange(1, n - k + 1))) - np.sum(np.log(np.arange(1, k + 1)))
    return coefficient

def binomial_pmf(k, n, p):
    coefficient = binomial_coefficient_log(k, n)
    prob = coefficient + k * np.log(p) + (n - k) * np.log(1-p)

    return np.exp(prob)

print("XXXXXXXXXXXX Binomial PMF XXXXXXXXXXXX")
print("k=10, n=1000, p=0.01")
print(binomial_pmf(10, 1000, 0.01))

def poisson_pmf(k, lam):
    logspace = k * np.log(lam) - np.sum(np.log(np.arange(1, k + 1))) + -lam
    return np.exp(logspace)


print("XXXXXXXXXXXX Poisson PMF XXXXXXXXXXXX")
print("k=10, n=1000, p=0.01")
print(poisson_pmf(10, 10))

print("XXXXXXXXXXXXXXXX Convergence XXXXXXXXXXXXXX")


def convergence(lam, k_max):
    k_vals = np.arange(1, k_max + 1)
    n_values = [10, 50, 200, 1000]
    colors = ['#4a90d9', '#50c0a0', '#e0c050', '#d06060']
    fig, ax = plt.subplots(figsize=(10, 5))

    for n, colour in zip(n_values, colors):
        pmf = [binomial_pmf(k, n, lam / n) for k in k_vals]
        ax.plot(k_vals, pmf, 'o--', color=colour, alpha=0.6,
                label=f'Binomial(n={n})', markersize=4)

    poisson = [poisson_pmf(k, lam) for k in k_vals]
    ax.plot(k_vals, poisson, 'k-', linewidth=2,
                label=f'Poisson(λ={lam})', zorder=10)
    ax.set_xlabel('k')
    ax.set_ylabel('P(X=k)')
    ax.set_title(f'Binomial → Poisson convergence (λ={lam})')
    ax.legend()
    plt.tight_layout()
    plt.show()

convergence(4, 15)

print("XXXXXXXXXXXXXX Simulate Poisson XXXXXXXXXXXXX")

def simulate_poisson(lam, n_samples):
    return np.random.binomial(n_samples, lam / n_samples, size=n_samples)

def plot_simulation(lam, n_samples):
    samples = simulate_poisson(lam, n_samples)
    k_max = 15
    k_vals = np.arange(0, k_max + 1)
    counts = np.bincount(samples[samples <= k_max], minlength=k_max + 1)
    empirical = counts / n_samples

    poisson = [poisson_pmf(k, lam) for k in k_vals]

    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.35
    ax.bar(k_vals - width/2, empirical, width,
           label='Simulated (via Binomial)', color='#4a90d9', alpha=0.7)
    ax.bar(k_vals + width/2, poisson, width,
           label='True Poisson PMF', color='#e06060', alpha=0.7)
    ax.set_title(f'Simulated vs True Poisson(λ={lam}), n={n_samples:,} samples')
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_simulation(4, 10000)

print("XXXXXXXXXX L1 Distance XXXXXXXXXXXX")

def l1_distance(lam, k_max):
    n_vals = [10, 50, 100, 500, 1000, 5000]
    k_vals = np.arange(1, k_max + 1)
    distances = []

    poisson = np.array([poisson_pmf(k, lam) for k in k_vals])
    for n in n_vals:
        binomial = np.array([binomial_pmf(k, n, lam / n) for k in k_vals])
        l1_dist = np.sum(np.abs(binomial - poisson))
        distances.append(l1_dist)

    print(distances)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xscale('log')
    ax.plot(n_vals, distances, '-', linewidth=2,
                label='Distances', zorder=10)
    ax.set_xlabel('n')
    ax.set_ylabel('Diff')
    ax.legend()
    plt.tight_layout()
    plt.show()
    return

l1_distance(4, 20)