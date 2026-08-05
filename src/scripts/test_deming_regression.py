import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import paths
from stats import median_standard_error, bootstrap_standard_error, deming_regression

fig, axs = plt.subplots(2,2, figsize=(8, 8), sharex=True, sharey=True)
xarr = np.arange(0, 8.1, 0.1)
abund_error_scale = 3

# Mock data
rng = np.random.default_rng()
true_params = [-0.08, 0.5]
xtrue = rng.uniform(low=0, high=8, size=5000)
ytrue = true_params[0] * xtrue + true_params[1]
sigma_x = 2
sigma_y = 0.05
xobs = xtrue + sigma_x * rng.standard_normal(size=xtrue.shape)
yobs = ytrue + sigma_y * rng.standard_normal(size=ytrue.shape)
xpos = xobs > 0
xobs = xobs[xpos]
yobs = yobs[xpos]
xerr = sigma_x * np.ones(xobs.shape)
yerr = sigma_y * np.ones(yobs.shape)
# print(true_params)

dem_reg = deming_regression(xobs, yobs, xerr, yerr)
axs[0,0].scatter(xobs, yobs, marker='.', s=1, c='gray')
axs[0,0].plot(xarr, true_params[0] * xarr + true_params[1], 'k-', label='True')
axs[0,0].plot(xarr, dem_reg[0] * xarr + dem_reg[1], 'r-', label='Deming')
axs[0,0].legend()
axs[0,0].set_title('Mock, accurate errors')
# print(dem_reg)
# print(bootstrap_standard_error(deming_regression, xobs, yobs, xerr, yerr))

# Plot mock data but with underestimated y-axis errors
sigma_y *= abund_error_scale
xobs = xtrue + sigma_x * rng.standard_normal(size=xtrue.shape)
yobs = ytrue + sigma_y * rng.standard_normal(size=ytrue.shape)
xpos = xobs > 0
xobs = xobs[xpos]
yobs = yobs[xpos]
xerr = sigma_x * np.ones(xobs.shape)
yerr = sigma_y/abund_error_scale * np.ones(yobs.shape)

dem_reg = deming_regression(xobs, yobs, xerr, yerr)
axs[1,0].scatter(xobs, yobs, marker='.', s=1, c='gray')
axs[1,0].plot(xarr, true_params[0] * xarr + true_params[1], 'k-', label='True')
axs[1,0].plot(xarr, dem_reg[0] * xarr + dem_reg[1], 'r-', label='Deming')
axs[1,0].set_title('Mock, underestimated y-errors')

# Import MWM data
sample = pd.read_csv(paths.data / 'sample.csv')
snsm = sample[
    (sample['Rg'] >= 7) &
    (sample['Rg'] < 9) &
    (sample['z_max'] < 0.5) &
    (sample['fe_h'] >= -0.1) &
    (sample['fe_h'] < 0.1) &
    (sample['good_age']) &
    (sample['high_ia'])
].copy()

# Parse data
xobs = snsm['age'].to_numpy()
yobs = snsm['ce_mg_corr'].to_numpy()
xerr = 0.5 * ((snsm['e_p_age'] - snsm['age']) + 
              (snsm['age'] - snsm['e_n_age'])).to_numpy()
yerr = snsm['e_ce_mg'].to_numpy()

# Plot data and regression
dem_reg = deming_regression(xobs, yobs, xerr, yerr)
axs[0,1].plot(xarr, dem_reg[0] * xarr + dem_reg[1], 'r-', label='Deming')
axs[0,1].scatter(xobs, yobs, marker='.', s=1, c='gray')
axs[0,1].set_title('MWM data')

# MWM but inflate the y-errors
yerr = yerr * abund_error_scale
dem_reg = deming_regression(xobs, yobs, xerr, yerr)
axs[1,1].plot(xarr, dem_reg[0] * xarr + dem_reg[1], 'r-', label='Deming')
axs[1,1].scatter(xobs, yobs, marker='.', s=1, c='gray')
axs[1,1].set_title('MWM data, inflated y-errors')
print(dem_reg)
print(bootstrap_standard_error(deming_regression, xobs, yobs, xerr, yerr))

for ax in axs[:,0]:
    ax.set_ylabel('[Ce/Mg]')
for ax in axs[-1]:
    ax.set_xlabel('Age [Gyr]')
axs[0,0].set_ylim((-0.7, 1))
plt.savefig(paths.extra / 'test_deming_regression.png')
plt.show()