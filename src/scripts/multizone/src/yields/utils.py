"""
Utility functions for multizone.src.yields
"""

from numbers import Number
import vice

class adjusted_agb(vice.yields.agb.interpolator): 
    """
    Provides for manual adjustments to the AGB yield grid for a given study. 
    Yields can be scaled or shifted in mass and metallicity space.

    Parameters
    ----------
    element : str
    study : str, optional [default: 'cristallo11']
    amp : float, optional [default: 1]
        Amplitude of AGB yields. If one, the yield scale is unchanged.
    dm : float, optional [default: 0]
        Linear shift to ZAMS mass of AGB progenitors. If positive, input
        mass is *decreased*, effectively increasing all masses in the grid.
    Zscale : float, optional [default: 1]
        Multiplicative shift to metallicity of AGB progenitors. If greater than 
        one, input metallicity is *decreased* by the given factor, effectively
        scaling up all metallicities in the grid.

    Inherits from vice.yields.agb.interpolator
    """
    def __init__(self, element, study='cristallo11', amp=1, dm=0, Zscale=1):
        self.amp = amp
        self.dm = dm
        self.Zscale = Zscale
        super().__init__(element, study=study)
    
    def __call__(self, mass, metallicity): 
        return max(
            self.amp * super().__call__(
                mass - self.dm, metallicity * 1 / self.Zscale
            ),
            0. # prevent negative yields from interpolation
        )
    
    @property
    def amp(self):
        """
        amp : float
            Amplitude of AGB yields. If one, the yield scale is unchanged.
        """
        return self._amp
    
    @amp.setter
    def amp(self, value):
        if isinstance(value, Number):
            if value > 0:
                self._amp = value
            else:
                raise ValueError('Yield amplitude must be positive.')
        else:
            raise TypeError(f'Parameter "amp" must be numeric, got: {type(value)}')
    
    @property
    def dm(self):
        """
        dm : float
            Linear shift to ZAMS mass of AGB progenitors. If positive, input
            mass is *decreased*, effectively increasing all masses in the grid.
        """
        return self._dm
    
    @dm.setter
    def dm(self, value):
        if isinstance(value, Number):
            self._dm = value
        else:
            raise TypeError(f'Parameter "dm" must be numeric, got: {type(value)}')
    
    @property
    def Zscale(self):
        """
        Zscale : float
            Multiplicative shift to metallicity of AGB progenitors. If greater 
            than one, input metallicity is *decreased* by the given factor,
            effectively scaling up all metallicities in the grid.
        """
        return self._Zscale
    
    @Zscale.setter
    def Zscale(self, value):
        if isinstance(value, Number):
            if value > 0:
                self._Zscale = value
            else:
                raise ValueError('Zscale must be positive.')
        else:
            raise TypeError(f'Parameter "Zscale" must be numeric, got: {type(value)}')
