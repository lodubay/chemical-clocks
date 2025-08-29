"""
Generate the Milky Way Mapper sample with quality cuts.
"""

import numpy as np
import pandas as pd
from astropy.io import fits
import astropy.units as u
import astropy.coordinates as coords

from utils import fits_to_pandas
import paths

def main():
    # Import full DR19 catalog (takes a while)
    print('Importing DR19 catalog...')
    mwm_full = fits_to_pandas(
        paths.data / 'MWM' / 'astraAllStarASPCAP-0.6.0.fits.gz', 
        hdu=2
    )
    # Quality cuts
    print('Implementing quality cuts...')
    mwm_good = mwm_full[
        (mwm_full['sdss4_apogee_extra_target_flags'] == 0) &
        (mwm_full['flag_bad'] == 0) & 
        (mwm_full['spectrum_flags'] == 0) &
        (mwm_full['snr'] > 40) &
        (mwm_full['sdss_id'] > 0)
    ].copy()
    # drop duplicate SDSS-V IDs with the lowest SNR
    mwm_good.sort_values(['sdss_id', 'snr'], inplace=True, ascending=True)
    mwm_good.drop_duplicates(subset='sdss_id', keep='last', inplace=True)
    # drop stars with no [Fe/H] or [O/H]
    mwm_good = mwm_good[
        (mwm_good['ce_h'] > -999) & 
        (mwm_good['mg_h'] > -999) &
        (mwm_good['fe_h'] > -999)
    ]
    mwm_good = mwm_good[
        (mwm_good['ce_h_flags'] == 0) &
        (mwm_good['mg_h_flags'] == 0) &
        (mwm_good['fe_h_flags'] == 0)
    ]
    # Calculate abundance ratios and errors in quadrature
    print('Calculating abundance ratios and coordinates...')
    mwm_good['mg_fe'], mwm_good['e_mg_fe'] = abundance_ratio(mwm_good, 'mg', 'fe')
    mwm_good['ce_mg'], mwm_good['e_ce_mg'] = abundance_ratio(mwm_good, 'ce', 'mg')
    mwm_good['c_n'], mwm_good['e_c_n'] = abundance_ratio(mwm_good, 'c', 'n')
    # Calculate galactocentric coordinates based on galactic l, b and Gaia dist
    galr, galphi, galz = galactic_to_galactocentric(
        mwm_good['l'], mwm_good['b'], mwm_good['r_med_photogeo']/1000
    )
    mwm_good['gal_r'] = galr # kpc
    mwm_good['gal_phi'] = galphi # deg
    mwm_good['gal_z'] = galz # kpc

    # Red giants only
    mwm_rgb = mwm_good[
        (mwm_good['logg'] > 1.0) & (mwm_good['logg'] < 3.7) &
        (mwm_good['teff'] < 5500) & (mwm_good['teff'] > 3500)
    ].copy()

    # Export catalogs
    print('Exporting full quality sample (MWM_good.csv)...')
    mwm_good.to_csv(paths.data / 'MWM' / 'MWM_good.csv', index=False)
    print('Exporting RGB sample (MWM_RGB.csv)...')
    mwm_rgb.to_csv(paths.data / 'MWM' / 'MWM_RGB.csv', index=False)
    print('Done!')


def abundance_ratio(catalog, elem1, elem2='fe_h'):
    """
    Compute element abundance ratios and errors in quadrature.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Full catalog with abundances.
    elem1 : str
        Numerator element, e.g. 'mg' or 'mg_h'. If no reference element is
        given, it is assumed to be relative to H, e.g. [Mg/H].
    elem2 : str, optional [default: 'fe_h']
        Denominator element. The default is 'fe_h'.
    
    Returns
    -------
    ratio : pandas.Series
        Element abundance ratio, e.g. [Mg/Fe].
    error : pandas.Series
        Error on the abundance ratio, summed in quadrature from individual
        uncertainties.
    """
    # Fill in implicit reference element
    if len(elem1) < 3 and '_' not in elem1:
        elem1 = f'{elem1}_h'
    if len(elem2) < 3 and '_' not in elem2:
        elem2 = f'{elem2}_h'
    ratio = catalog[elem1] - catalog[elem2]
    error = np.sqrt(catalog[f'e_{elem1}']**2 + catalog[f'e_{elem2}']**2)
    return ratio, error


def galactic_to_galactocentric(l, b, distance):
    r"""
    Use astropy's SkyCoord to convert Galactic (l, b, distance) coordinates
    to galactocentric (r, phi, z) coordinates.

    Parameters
    ----------
    l : array-like
        Galactic longitude in degrees
    b : array-like
        Galactic latitude in degrees
    distance : array-like
        Distance (from Sun) in kpc

    Returns
    -------
    galr : numpy array
        Galactocentric radius in kpc
    galphi : numpy array
        Galactocentric phi-coordinates in degrees
    galz : numpy arraay
        Galactocentric z-height in kpc
    """
    l = np.array(l)
    b = np.array(b)
    d = np.array(distance)
    if l.shape == b.shape == d.shape:
        if not isinstance(l, u.quantity.Quantity):
            l *= u.deg
        if not isinstance(b, u.quantity.Quantity):
            b *= u.deg
        if not isinstance(d, u.quantity.Quantity):
            d *= u.kpc
        galactic = coords.SkyCoord(l=l, b=b, distance=d, frame=coords.Galactic())
        galactocentric = galactic.transform_to(frame=coords.Galactocentric())
        galactocentric.representation_type = 'cylindrical'
        galr = galactocentric.rho.to(u.kpc).value
        galphi = galactocentric.phi.to(u.deg).value
        galz = galactocentric.z.to(u.kpc).value
        return galr, galphi, galz
    else:
        raise ValueError('Arrays must be of same length.')
    

if __name__ == '__main__':
    main()
