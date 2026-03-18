"""
This file implements the equilibrium-based yields of Weinberg et al. (2024).
"""

import vice

# Massive star explosion fraction
Fexp = 0.75
# Mean CCSN Fe yield (Msun)
mfecc = 0.058
# Mean SN Ia Fe yield (Msun)
mfeia = 0.7
# Rate of SNe Ia per solar mass of stars
Ria = 1.3e-3 # Maoz & Graur (2017)
# [alpha/Fe] plateau value for pure-alpha elements
afecc = 0.45

def ccsn_ratio(Fexp=0.75, Mmin=0.08, Mmax=120, Mthresh=8, dm=0.01, 
               imf=vice.imf.kroupa):
    r"""
    Calculate the core-collapse supernova ratio per total mass of stars.
    
    Parameters
    ----------
    Fexp : float, optional
        Fraction of massive stars which explode. The default is 0.75.
    Mmin : float, optional
        Minimum stellar mass for IMF integration. The default is 0.08.
    Mmax : float, optional
        Maximum stellar mass for IMF integration. The default is 120.
    Mthresh : float, optional
        Minimum mass of stars that explode as core-collapse supernovae.
        The default is 8.
    dm : float, optional
        Integration mass step size. The default is 0.1.
    imf : function, optional
        The initial mass function dN/dm. Must accept a single argument,
        which is stellar mass. The default is a Kroupa IMF.
        
    Returns
    -------
    float
        The core-collapse supernova ratio $R_{\rm cc}$.
    """
    # Integration masses
    masses = [m * dm + Mmin for m in range(int((Mmax + dm - Mmin) / dm))]
    N_massive_stars = sum([imf(m) * dm for m in masses if m >= Mthresh])
    total_mass = sum([m * imf(m) * dm for m in masses])
    return Fexp * N_massive_stars / total_mass

# IMF-averaged CCSN yields
# yield calibration is based on Weinberg++ 2024, eq. 10
Rcc = ccsn_ratio(Fexp=Fexp) # CCSNe per unit stellar mass
mmgcc = mfecc * 10 ** afecc * vice.solar_z["mg"] / vice.solar_z["fe"]
vice.yields.ccsne.settings["mg"] = Rcc * mmgcc
vice.yields.ccsne.settings["fe"] = Rcc * mfecc

# population averaged SNIa Fe yield
vice.yields.sneia.settings["fe"] = Ria * mfeia
# Other SN Ia element yields
vice.yields.sneia.settings["mg"] = 0.
