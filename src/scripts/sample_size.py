"""
Save tex output files with various sample sizes.
"""

import paths
from utils import import_sample

sample = import_sample(good_ages=False, cut_limits=False)

# Stars with [Ce/H] upper limits
with open(paths.output / 'upper_limits.tex', 'w') as f:
    f.write('\\num{%s}' % str(sample[sample['lim_ce_h_flag'] > 0].shape[0]))

# Full RGB sample
sample = sample[sample['lim_ce_h_flag'] == 0]
with open(paths.output / 'sample_size.tex', 'w') as f:
    f.write('\\num{%s}' % str(sample.shape[0]))

# Good ages
with open(paths.output / 'good_ages.tex', 'w') as f:
    f.write('\\num{%s}' % str(sample[sample['good_age']].shape[0]))
