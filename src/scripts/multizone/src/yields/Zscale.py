"""
Yields with Ce production shifted to higher-metallicity AGB stars.
"""

import vice
from .utils import adjusted_agb

# Use solar-scaled CCSN yields from Weinberg et al. (2024)
from . import W24

# AGB yields from Cristallo et al. (2011, 2015)
vice.yields.agb.settings["ce"] = adjusted_agb(
    "ce", 
    study="cristallo11",
    amp=1,
    dm=0,
    Zscale=2, # shift production to higher-metallicity progenitors
)

# Residual fraction of Solar Ce produced by r-process
Fcer = 0.23 # Arlandini et al. (1999)
# Assign r-process Ce production to CCSN channel
vice.yields.ccsne.settings["ce"] = Fcer * (
    vice.yields.ccsne.settings["mg"] * vice.solar_z["ce"] / vice.solar_z["mg"]
)
vice.yields.sneia.settings["ce"] = 0.
