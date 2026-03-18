"""
Utility functions for multizone.src.yields
"""

from numbers import Number
from vice.toolkit.interpolation.interp_scheme_2d import interp_scheme_2d
from vice.yields.agb._grid_reader import yield_grid


class agb_interpolator(interp_scheme_2d):
    """
    Custom AGB yield interpolator that forces 0 yield at 0 metallicity,
    0 yield at 0 mass, and enforces non-negative yields.
    
    Inherits from vice.toolkit.interpolator.interp_scheme_2d.
    """
    def __init__(self, element, study='cristallo11'):
        # let the grid reader function do the error handling
        yields, masses, metallicities = yield_grid(element, study=study)
        # enforce yield of 0 at 0 mass
        new_masses = [0] + list(masses)
        # enforce yield of 0 at 0 metallicity
        new_metallicities = [0] + list(metallicities)
        new_yields = [[0] * len(new_metallicities)]
        for row in yields:
            new_yields.append([0] + list(row))
        super().__init__(new_masses, new_metallicities, new_yields)
    
    def __call__(self, mass, metallicity):
        return max(super().__call__(mass, metallicity), 0)
    
    @property
    def masses(self):
        return super().xcoords
    
    @property
    def metallicities(self):
        return super().ycoords
    
    @property
    def yields(self):
        return super().zcoords


class adjusted_agb(agb_interpolator): 
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
        Cannot be used with mscale.
    mscale : float, optional [default: 1]
        Multiplicative shift to the ZAMS mass of AGB progenitors. If greater
        than one, input mass is *decreased* by the given factor, effectively
        scaling up all masses in the grid. Cannot be used with dm.
    Zscale : float, optional [default: 1]
        Multiplicative shift to metallicity of AGB progenitors. If greater than 
        one, input metallicity is *decreased* by the given factor, effectively
        scaling up all metallicities in the grid.

    Inherits from agb_interpolator
    """
    def __init__(self, element, study='cristallo11', amp=1, dm=0, mscale=1, Zscale=1):
        if dm != 0 and mscale != 1:
            raise ValueError('Both dm and mscale have been modified.')
        self.amp = amp
        self.dm = dm
        self.mscale = mscale
        self.Zscale = Zscale
        super().__init__(element, study=study)
    
    def __call__(self, mass, metallicity): 
        return self.amp * super().__call__(
            mass * 1 / self.mscale - self.dm, metallicity * 1 / self.Zscale
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
    def mscale(self):
        """
        mscale : float
            Multiplicative shift to ZAMS mass of AGB progenitors. If greater
            than one, mass is *decreased*, effectively increasing all masses 
            in the grid.
        """
        return self._mscale
    
    @mscale.setter
    def mscale(self, value):
        if isinstance(value, Number):
            if value > 0:
                self._mscale = value
            else:
                raise ValueError('Parameter "mscale" must be positive.')
        else:
            raise TypeError(f'Parameter "mscale" must be numeric, got: {type(value)}')
    
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
        

class decompose_agb_grid:
    """
    Allows for an AGB yield grid to be broken into multiple components,
    with each component individually scaled and shifted in mass or metallicity.

    This class is designed some fraction of the AGB yield to be from
    blue stragglers (i.e., AGB stars with a ZAMS mass of 2 Msun but an age
    of a 1 Msun star).
    """
    def __init__(
            self, element, study='cristallo11', 
            amplitudes=1, mshifts=0, mscales=1, Zscales=1
        ):
        params = [amplitudes, mshifts, mscales, Zscales]
        iterables = [p for p in params if isinstance(p, list)]
        # Error handling: ensure iterable parameters are same length
        if same_length(iterables):
            ncomponents = len(iterables[0])
            # Turn non-iterable parameters into lists
            for i, param in enumerate(params):
                if not isinstance(param, list):
                    params[i] = [param] * ncomponents
        else:
            raise ValueError('If multiple parameters are iterable, all must \
have the same length.')
        # Each component is an adjusted_agb instance with its own amplitude,
        # mass and metallicity scalings
        self.components = [
            adjusted_agb(
                element, 
                study=study, 
                amp=params[0][i], 
                dm=params[1][i], 
                mscale=params[2][i], 
                Zscale=params[3][i]
            ) for i in range(ncomponents)
        ]

    def __call__(self, mass, metallicity):
        return sum([comp(mass, metallicity) for comp in self.components])

    @property
    def components(self):
        """
        list of adjusted_agb instances
        """
        return self._components
    
    @components.setter
    def components(self, value):
        try:
            if all([isinstance(c, adjusted_agb) for c in value]):
                self._components = value
            else:
                raise TypeError('All components must be instances of adjusted_agb')
        except:
            raise TypeError('Components must be iterable! Got: %s' % type(value))

    @property
    def ncomponents(self):
        """
        ncomponents : int
            Number of individual AGB grid components.
        """
        return len(self.components)


def same_length(items):
    """
    Checks if all the items in list x are lists with the same length.
    """
    return all(len(x) == len(items[0]) for x in items)
