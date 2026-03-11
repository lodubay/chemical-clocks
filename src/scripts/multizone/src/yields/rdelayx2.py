"""
Yields with a delayed r-process component via the SN Ia channel.
"""

import vice
from .utils import adjusted_agb
from ..._globals import SOLAR_CE_S_FRAC

# Use solar-scaled CCSN yields from Weinberg et al. (2024)
from . import W24

# AGB yields from Cristallo et al. (2011, 2015)
vice.yields.agb.settings["ce"] = adjusted_agb(
    "ce", 
    study="cristallo11",
    amp=1,
    mscale=1,
    Zscale=1,
)

# Residual fraction of Solar Ce produced by r-process
Fcer = 1 - SOLAR_CE_S_FRAC # Arlandini et al. (1999)
# Assign r-process Ce production to SNIa channel
vice.yields.ccsne.settings["ce"] = 0.
vice.yields.sneia.settings["ce"] = 2 * Fcer * (
    vice.yields.ccsne.settings["mg"] * vice.solar_z["ce"] / vice.solar_z["mg"]
)
