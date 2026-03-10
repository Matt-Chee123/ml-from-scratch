import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n = 10_000

X = rng.uniform(-1, 1, n)
Y = X ** 2

#E[X]
print("E[X] - ",np.mean(X))

#E[Y]
print("E[Y] - ",np.mean(Y))

#E[XY]
print("E[XY] - ", np.mean(X * Y))

#Cov(X,Y)
print("Cov(X,Y) - ", np.cov(X, Y)[0,1])

#Corr(X, Y)
print("Corr(X,Y) - ", np.corrcoef(X, Y)[0, 1])

x_val = 0.7
y_if_x = x_val ** 0.2
print(f"\nIf X = {x_val}, then Y = {y_if_x} exactly.")
print("This is not consistent with independence.")

# Formal conditional distribution test
# if independent then P(Y > 0.3) = P(Y > 0.3 | X < 0.5)
x_mask = X < 0.6
y1 = np.mean(Y > 0.3)
y2 = np.mean(Y[x_mask] > 0.3)
print("P(Y > 0.3) - ", y1)
print("P(Y > 0.3 | X < 0.5) - ", y2)

fig, axes = plt.subplots(1, 3, figsize=(15,4))

axes[0].scatter(X[:2000], Y[:2000], alpha=0.2, s=8, color='steelblue')
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y = X²")
axes[0].set_title(f"Scatter: Y = X²\nCov = {np.cov(X,Y)[0,1]:.5f}")

axes[1].hist(X, bins=60, alpha=0.6, label="X", density=True, color='steelblue')
axes[1].hist(Y, bins=60, alpha=0.6, label="Y", density=True, color='coral')
axes[1].set_title("Marginal distributions\n(give no hint of the dependence)")
axes[1].legend()

bin_edges = np.linspace(-1, 1, 30)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
cond_means = [Y[(X >= bin_edges[i]) & (X < bin_edges[i+1])].mean()
              for i in range(len(bin_edges)-1)]
axes[2].plot(bin_centres, cond_means, 'o-', color='coral', markersize=4)
axes[2].plot(bin_centres, bin_centres**2, '--', color='black',
             label='True X²', linewidth=2)
axes[2].set_xlabel("X")
axes[2].set_ylabel("E[Y | X]")
axes[2].set_title("Conditional mean E[Y|X]\nshows strong dependence")
axes[2].legend()

plt.tight_layout()
plt.show()

