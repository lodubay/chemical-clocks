"""
Yields for the multizone models adopting the Karakas & Lugaro (2016)
AGB yields, with scaled masses.
"""

import vice
from .utils import adjusted_agb

# Use solar-scaled CCSN yields from Weinberg et al. (2024)
from . import W24
# Double the SN yields
vice.yields.ccsne.settings["mg"] *= 2
vice.yields.ccsne.settings["fe"] *= 2
vice.yields.sneia.settings["fe"] *= 2

# AGB yields from Karakas & Lugaro (2016); Karakas et al. (2018)
vice.yields.agb.settings["ce"] = adjusted_agb(
    "ce", 
    study="karakas16",
    amp=1,
    mscale=0.5, # shift production to lower-mass AGB progenitors
    Zscale=1,
)

# Residual fraction of Solar Ce produced by r-process
Fcer = 0.23 # Arlandini et al. (1999)
# Assign r-process Ce production to CCSN channel
vice.yields.ccsne.settings["ce"] = Fcer * (
    vice.yields.ccsne.settings["mg"] * vice.solar_z["ce"] / vice.solar_z["mg"]
)
vice.yields.sneia.settings["ce"] = 0.
