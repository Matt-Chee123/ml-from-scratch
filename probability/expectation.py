import numpy as np
np.random.seed(42)
N = 5_000_000

X = np.random.normal(0, 1, N)
Y = np.random.normal(3, 2, N)
a, b = 4, 7

# Verify E[aX + b] = aE[X] + b

lhs = np.mean(a * X + b)
rhs = a * np.mean(X) + b
print("LHS: ", lhs, " RHS: ", rhs)

# Verify E[X + Y] = E[X] + E[Y]  (X, Y independent here)

lhs = np.mean(X + Y)
rhs = np.mean(X) + np.mean(Y)
print("LHS: ", lhs, " RHS: ", rhs)

# Verify E[XY] = E[X]E[Y] when independent

lhs = np.mean(X * Y)
rhs = np.mean(X) * np.mean(Y)
print("LHS: ", lhs, " RHS: ", rhs)

# Now break independence — Y2 = X^2
Y2 = X ** 2 + X
lhs = np.mean(X * Y2)
rhs = np.mean(X) * np.mean(Y2)
print("LHS: ", lhs, " RHS: ", rhs)