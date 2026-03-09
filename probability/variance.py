import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
X = rng.normal(loc=3, scale=2, size=10_000)

# E[X^2] - (E[X])^2 = Var(X)

lhs = np.mean(X ** 2) - np.mean(X) ** 2
rhs = np.var(X)

print("# E[X^2] - (E[X])^2 = Var(X)")
print("LHS - ", lhs, " RHS - ", rhs)

# Var(aX + b) = a^2Var(X)
a1, a2, a3 = 1, 3, 3
b1, b2, b3 = 10, 0, 100

lhs1 = np.var(a1 * X + b1)
lhs2 = np.var(a2 * X + b2)
lhs3 = np.var(a3 * X + b3)

rhs1 = a1 ** 2 * np.var(X)
rhs2 = a2 ** 2 * np.var(X)
rhs3 = a3 ** 2 * np.var(X)

print("Var(aX + b) = a^2Var(X)")
print("LHS1 - ", lhs1, " RHS1 - ", rhs1)
print("LHS2 - ", lhs2, " RHS2 - ", rhs2)
print("LHS3 - ", lhs3, " RHS3 - ", rhs3)

#Var(X+Y) = Var(X) + Var(Y) + 2·Cov(X,Y)
YCorr = 0.5 * X + rng.normal(0, 1, size=10_000)

lhs = np.var(X + YCorr)
wrongRhs = np.var(X) + np.var(YCorr)
covariance = np.cov([X, YCorr])[0, 1]
rhs = np.var(X) + np.var(YCorr) + 2 * covariance
print("Correlated Var(X+Y) = Var(X) + Var(Y) + 2·Cov(X,Y)")
print("LHS - ", lhs, " RHS - ", rhs," Covariance - ",covariance, " Wrong RHS - ", wrongRhs)


YUncorr = rng.normal(0, 1, size=10_000)
lhs = np.var(X + YUncorr)
uncorr = np.var(X) + np.var(YUncorr)
covariance = np.cov([X, YUncorr])[0, 1]
rhs = np.var(X) + np.var(YUncorr) + covariance
print("Unorrelated Var(X+Y) = Var(X) + Var(Y) + 2·Cov(X,Y)")
print("LHS - ", lhs, " RHS - ", rhs," Covariance - ",uncorr, " No Covariance RHS - ", uncorr)

ns = [5, 10, 50, 100, 500, 1000, 5000, 10000]
true_var = 4.0

plt.figure(figsize=(9, 5))

for x in range(1):
    estimates = [np.var(rng.normal(3,2, size=n)) for n in ns]
    plt.plot(ns, estimates, alpha=0.3, color='steelblue', linewidth=1)

plt.axhline(true_var, color='tomato', linestyle='--', linewidth=1.5, label='True variance = 4')

plt.xscale('log')
plt.xlabel('Sample size (n)')
plt.ylabel('Estimated variance')
plt.title('Sample variance converges to true variance as n → ∞')
plt.legend()
plt.tight_layout()
plt.show()