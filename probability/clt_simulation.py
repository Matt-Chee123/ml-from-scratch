import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def simulate_clt(dist_fn, n_obs, n_sims, true_mean, true_std):
    samples = dist_fn(n_obs * n_sims).reshape(n_sims, n_obs)

    sample_means = samples.mean(axis=1)

    standard_error = true_std / np.sqrt(n_obs)

    z_scores = (sample_means - true_mean) / standard_error
    return z_scores


rng = np.random.default_rng(seed=42)

distributions = {
    "Uniform(0,1)": {
        "fn"   : lambda n: rng.uniform(0, 1, n),
        "mean" : 0.5,
        "std"  : 1 / np.sqrt(12),
        "color": "steelblue",
    },
    "Exponential(1)": {
        "fn"   : lambda n: rng.exponential(scale=1.0, size=n),
        "mean" : 1.0,
        "std"  : 1.0,
        "color": "coral",
    },
    "Bernoulli(0.3)": {
        "fn"   : lambda n: rng.binomial(1, 0.3, n).astype(float),
        "mean" : 0.3,
        "std"  : np.sqrt(0.3 * 0.7),
        "color": "mediumseagreen",
    },
}

n_values = [1, 5, 30]
n_sims   = 10_000

results = {}
for name, config in distributions.items():
    results[name] = {}
    for n in n_values:
        results[name][n] = simulate_clt(
            dist_fn   = config["fn"],
            n_obs     = n,
            n_sims    = n_sims,
            true_mean = config["mean"],
            true_std  = config["std"],
        )

dist_names = list(distributions.keys())
x_range    = np.linspace(-4, 4, 200)
normal_pdf = norm.pdf(x_range)

fig, axes = plt.subplots(3, 3, figsize=(12, 9))
fig.suptitle(
    "CLT Simulation — Standardised Sample Means vs N(0,1)\n"
    "(red curve = N(0,1), should match histogram as n grows)",
    fontsize=13, y=1.02
)

for row, dist_name in enumerate(dist_names):
    color = distributions[dist_name]["color"]

    for col, n in enumerate(n_values):
        ax = axes[row][col]
        z  = results[dist_name][n]

        ax.hist(z, bins=60, density=True, alpha=0.75,
                color=color, edgecolor="none")

        ax.plot(x_range, normal_pdf, color="crimson",
                linewidth=2, label="N(0,1)")

        ax.set_xlim(-4, 4)
        ax.set_xlabel("z-score", fontsize=8)
        ax.tick_params(labelsize=8)

        if row == 0:
            ax.set_title(f"n = {n}", fontsize=11, fontweight="bold")

        if col == 0:
            ax.set_ylabel(dist_name, fontsize=9)

plt.tight_layout()
plt.show()

exp_config  = distributions["Exponential(1)"]
n_extended  = [1, 5, 30, 100]

exp_results = {
    n: simulate_clt(
        dist_fn   = exp_config["fn"],
        n_obs     = n,
        n_sims    = n_sims,
        true_mean = exp_config["mean"],
        true_std  = exp_config["std"],
    )
    for n in n_extended
}

fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
fig2.suptitle(
    "Berry-Esseen: Exponential(1) — skewness=2 slows CLT convergence\n"
    "n=30 still right-skewed. n=100 nearly indistinguishable from N(0,1).",
    fontsize=12
)

for i, n in enumerate(n_extended):
    ax = axes2[i]
    ax.hist(exp_results[n], bins=60, density=True,
            alpha=0.75, color="coral", edgecolor="none")
    ax.plot(x_range, normal_pdf, color="crimson", linewidth=2)
    ax.set_title(f"n = {n}", fontsize=11, fontweight="bold")
    ax.set_xlim(-4, 4)
    ax.set_xlabel("z-score", fontsize=9)
    ax.tick_params(labelsize=8)

    be_error = 2 / np.sqrt(n)
    ax.text(0.97, 0.97, f"BE≈{be_error:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="dimgray")

axes2[0].set_ylabel("Density", fontsize=9)
plt.tight_layout()
plt.show()
