"""
Yields with some Ce production shifted to lower masses.

Assumes 50% of total yields come from blue stragglers and other merger products
(i.e., the yield of a 2 Msun AGB, but the lifetime of a 1 Msun star).
"""

import vice
from .utils import decompose_agb_grid

# Use solar-scaled CCSN yields from Weinberg et al. (2024)
from . import W24

# AGB yields from Cristallo et al. (2011, 2015)
vice.yields.agb.settings["ce"] = decompose_agb_grid(
    "ce", 
    study="cristallo11",
    amplitudes=[0.5, 0.5],
    mscales=[1, 0.5],
    mshifts=0,
    Zscales=1,
)

# Residual fraction of Solar Ce produced by r-process
Fcer = 0.23 # Arlandini et al. (1999)
# Assign r-process Ce production to CCSN channel
vice.yields.ccsne.settings["ce"] = Fcer * (
    vice.yields.ccsne.settings["mg"] * vice.solar_z["ce"] / vice.solar_z["mg"]
)
vice.yields.sneia.settings["ce"] = 0.
