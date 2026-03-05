"""
AGB yields for Ce only (no r-process).
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
    Zscale=1,
)

vice.yields.ccsne.settings["ce"] = 0.
vice.yields.sneia.settings["ce"] = 0.
