import numpy as np
from scipy import stats

def mle_gaussian(data):
    mu_hat = np.sum(data) / len(data)
    sigma_sq_hat = np.sum((data - mu_hat)**2) / len(data)

    return (mu_hat, sigma_sq_hat)

def mle_bernoulli(data):
    p_hat = sum(data) / len(data)
    return p_hat

normaldata = np.random.normal(3.0, 2.0, 1000)
bernoulli_data = np.random.binomial(1, 0.7, 1000)
mu_hat, sigma_hat = mle_gaussian(normaldata)
p_hat = mle_bernoulli(bernoulli_data)


print(f"Gaussian  →  μ̂ = {mu_hat:.4f}  (true: 3.0)")
print(f"          →  σ̂² = {sigma_hat:.4f}  (true: 4.0)")
print(f"Bernoulli →  p̂ = {p_hat:.4f}  (true: 0.7)")

scipy_mu, scipy_sigma = stats.norm.fit(normaldata)
scipy_sig_sq = scipy_sigma ** 2

print(f"\nCross-check (should match to 4dp):")
print(f"  μ̂   ours={mu_hat:.6f}  scipy={scipy_mu:.6f}  match={np.isclose(mu_hat, scipy_mu)}")
print(f"  σ̂²  ours={sigma_hat:.6f}  scipy={scipy_sig_sq:.6f}  match={np.isclose(sigma_hat, scipy_sig_sq)}")
