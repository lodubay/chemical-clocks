import numpy as np
import pandas as pd

import paths
from stats import median_standard_error, bootstrap_standard_error, deming_regression

# sample = pd.read_csv(paths.data / 'MWM/sample.csv')
# snsm = sample[
#     (sample['Rg'] >= 7) &
#     (sample['Rg'] < 9) &
#     (sample['z_max'] < 0.5) &
#     (sample['fe_h'] >= -0.1) &
#     (sample['fe_h'] < 0.1) &
#     (sample['good_age'])
# ].copy()

# # Parse data
# xobs = snsm['age'].to_numpy()
# yobs = snsm['ce_mg_corr'].to_numpy()
# xerr = 0.5 * ((snsm['e_p_age'] - snsm['age']) + 
#               (snsm['age'] - snsm['e_n_age'])).to_numpy()
# yerr = snsm['e_ce_mg'].to_numpy()

rng = np.random.default_rng()
true_params = [-0.05, 0.5]
# xtrue = np.arange(0, 10.1, 0.1)
xtrue = rng.uniform(low=0, high=10, size=1000)
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

print(xtrue[:10])
print(xobs[:10])

print(true_params)
print(deming_regression(xobs, yobs, xerr, yerr))
print(bootstrap_standard_error(deming_regression, xobs, yobs, xerr, yerr))

# print(median_standard_error(xobs))
# print(bootstrap_standard_error(np.median, xobs))
