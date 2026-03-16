import numpy as np
import matplotlib.pyplot as plt

def bootstrap_ci(data, n_resamples=1000, ci=0.95):
    length = len(data)
    sample_means = []

    for _ in range(n_resamples):
        n_sample = np.random.choice(data, size=length, replace=True)
        sample_means.append(np.mean(n_sample))

    lower = (1 - ci) / 2
    higher = ci + lower
    intervals = np.percentile(sample_means, [lower * 100, higher * 100])
    print(lower, higher)
    print(sample_means)
    return {
        'mean': np.mean(sample_means),
        'lower': np.mean(intervals[0]),
        'upper': np.mean(intervals[1]),
        'all_means': sample_means}


data = np.random.normal(5, 2, size=100)
result = bootstrap_ci(data)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(result['all_means'], bins=40, color='#AFA9EC', edgecolor='white')
axes[0].axvline(result['lower'], color='#534AB7', linewidth=2, linestyle='--', label='95% CI')
axes[0].axvline(result['upper'], color='#534AB7', linewidth=2, linestyle='--')
axes[0].axvline(result['mean'],  color='#26215C', linewidth=2, label='bootstrap mean')
axes[0].set_title('Bootstrap sampling distribution')
axes[0].set_xlabel('Sample mean')
axes[0].legend()

axes[1].hist(data, bins=30, alpha=0.5, color='#5DCAA5', edgecolor='white', label='raw data')
axes[1].hist(result['all_means'], bins=40, alpha=0.7, color='#AFA9EC', edgecolor='white', label='bootstrap means')
axes[1].set_title('Raw data vs bootstrap means')
axes[1].set_xlabel('Value')
axes[1].legend()

true_mean = 5
n_trials = 100
colors, lowers, uppers = [], [], []

for _ in range(n_trials):
    d = np.random.normal(true_mean, 2, size=100)
    r = bootstrap_ci(d)
    lowers.append(r['lower'])
    uppers.append(r['upper'])
    colors.append('#1D9E75' if r['lower'] <= true_mean <= r['upper'] else '#E24B4A')

for i in range(n_trials):
    axes[2].plot([lowers[i], uppers[i]], [i, i], color=colors[i], linewidth=1.2, alpha=0.8)
axes[2].axvline(true_mean, color='#26215C', linewidth=1.5, linestyle='--', label=f'true mean = {true_mean}')
axes[2].set_title(f'Coverage: {colors.count("#1D9E75")}/100 CIs contain μ')
axes[2].set_xlabel('Value')
axes[2].set_yticks([])
axes[2].legend()

plt.tight_layout()
plt.show()