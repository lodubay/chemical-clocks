"""
Generate the Milky Way Mapper sample with quality cuts.
"""

import numpy as np
import pandas as pd
from astropy.io import fits
import astropy.units as u
import astropy.coordinates as coord
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from galpy.actionAngle import estimateDeltaStaeckel
import gala.dynamics as gd
import gala.potential as gp

from utils import fits_to_pandas
import paths

def main():
    # Import full DR19 catalog (takes a while)
    print('Importing DR19 catalog...')
    mwm_full = fits_to_pandas(
        paths.data / 'MWM' / 'astraAllStarASPCAP-0.6.0.fits.gz', 
        hdu=2
    )
    # Join row-matched StarFlow age catalog
    starflow = fits_to_pandas(
        paths.data / 'MWM' / 'StarFlow_summary_v1_0_1.fits'
    )
    # ensure SDSS IDs are the same between DR19 and StarFlow in every row
    assert np.all(np.where(mwm_full['sdss_id'] == starflow['sdss_id'], 1, 0))
    mwm_full = mwm_full.join(
        starflow[['age', 'e_p_age', 'e_n_age', 'training_density', 'BITMASK']]
    )
    # Quality cuts
    print('Implementing quality cuts...')
    mwm_good = mwm_full[
        (mwm_full['sdss4_apogee_extra_target_flags'] < 2) &
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
        # (mwm_good['ce_h_flags'] == 0) &
        (mwm_good['mg_h_flags'] == 0) &
        (mwm_good['fe_h_flags'] == 0)
    ]
    # Calculate abundance ratios and errors in quadrature
    print('Calculating abundance ratios and coordinates...')
    mwm_good['mg_fe'], mwm_good['e_mg_fe'] = abundance_ratio(mwm_good, 'mg', 'fe')
    mwm_good['ce_mg'], mwm_good['e_ce_mg'] = abundance_ratio(mwm_good, 'ce', 'mg')
    mwm_good['ce_fe'], mwm_good['e_ce_fe'] = abundance_ratio(mwm_good, 'ce', 'fe')
    mwm_good['c_n'], mwm_good['e_c_n'] = abundance_ratio(mwm_good, 'c', 'n')
    # Require Gaia distances
    mwm_good.dropna(axis=0, how='any', subset=['r_med_photogeo'], inplace=True)
    # Calculate galactocentric coordinates based on galactic l, b and Gaia dist
    galr, galphi, galz = galactic_to_galactocentric(
        mwm_good['l'], mwm_good['b'], mwm_good['r_med_photogeo']/1000
    )
    mwm_good['gal_r'] = galr # kpc
    mwm_good['gal_phi'] = galphi # deg
    mwm_good['gal_z'] = galz # kpc
    print('Joining with orbit parameters...')
    mwm_good = add_kinematics(mwm_good, id_name='gaia_dr3_source_id', verbose=True)
    # print('Computing orbits...')
    # rguide, zmax, ecc, energy, Lz = orbit_dynamics(
    #     mwm_good['ra'], mwm_good['dec'], mwm_good['r_med_photogeo']/1000,
    #     mwm_good['pmra'], mwm_good['pmde'], mwm_good['v_rad']
    # )
    # mwm_good['galpy_r_guide'] = rguide
    # mwm_good['galpy_z_max'] = zmax
    # mwm_good['galpy_ecc'] = ecc
    # mwm_good['galpy_E'] = energy
    # mwm_good['galpy_Lz'] = Lz
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
        # Define galactocentric coordinate frame
        with coord.galactocentric_frame_defaults.set('v4.0'):
            galcen_frame = coord.Galactocentric()
        galactic = coord.SkyCoord(l=l, b=b, distance=d, frame=coord.Galactic())
        galactocentric = galactic.transform_to(frame=galcen_frame)
        galactocentric.representation_type = 'cylindrical'
        galr = galactocentric.rho.to(u.kpc).value
        galphi = galactocentric.phi.to(u.deg).value
        galz = galactocentric.z.to(u.kpc).value
        return galr, galphi, galz
    else:
        raise ValueError('Arrays must be of same length.')
    

def orbit_dynamics(ra, dec, dist, pmra, pmdec, vrad, approx='staeckel'):
    """
    Integrate orbits and compute orbital dynamics with galpy.

    Parameters
    ----------
    ra : array-like
        Right ascension in degrees.
    dec : array-like
        Declination in degrees.
    dist : array-like
        Distance from Sun in kpc.
    pmra : array-like
        Proper motion in RA in mas/yr.
    pmdec : array-like
        Proper motion in Dec in mas/yr.
    vrad : array-like
        Radial velocity in km/s.
    approx : str, optional [default: 'staeckel']
        Type of analytic approximation to use. Passed to galpy methods.
    """
    ra = np.array(ra)
    dec = np.array(dec)
    dist = np.array(dist)
    pmra = np.array(pmra)
    pmdec = np.array(pmdec)
    vrad = np.array(vrad)
    args = [ra, dec, dist, pmra, pmdec, vrad]
    if [ra.shape == a.shape for a in args[1:]]:
        # Define galactocentric coordinate frame
        with coord.galactocentric_frame_defaults.set('v4.0'):
            galcen_frame = coord.Galactocentric()
        sky_coords = coord.SkyCoord(
            ra=ra*u.deg, 
            dec=dec*u.deg, 
            distance=dist*u.kpc,
            pm_ra_cosdec=pmra*u.mas/u.yr, 
            pm_dec=pmdec*u.mas/u.yr,
            radial_velocity=vrad*u.km/u.s,
            frame=coord.ICRS()
        )
        galcen_coords = sky_coords.transform_to(galcen_frame)
        orbits = Orbit(galcen_coords)
        galcen_coords.representation_type = 'cylindrical'
        delta = estimateDeltaStaeckel(
            MWPotential2014, 
            galcen_coords.rho, 
            galcen_coords.z, 
            no_median=True
        )
        kwargs = dict(pot=MWPotential2014, type=approx, delta=delta)
        rguide = orbits.rguiding(**kwargs)
        zmax = orbits.zmax(analytic=True, **kwargs)
        ecc = orbits.e(analytic=True, **kwargs)
        energies = orbits.E(pot=MWPotential2014)
        Lz = orbits.L(pot=MWPotential2014)[:,2]
        # stars_w0 = gd.PhaseSpacePosition(galcen_coords.data)
        # mw_potential = gp.MilkyWayPotential()
        # stars_orbit = mw_potential.integrate_orbit(stars_w0, dt=0.5 * u.Myr, t1=0, t2=4 * u.Gyr, cython_if_possible=True)
        # rguide = stars_orbit[0].guiding_radius().value
        # zmax = stars_orbit.zmax().value
        # ecc = stars_orbit.eccentricity()
        # energies = stars_orbit.energy()[0].to(u.km**2/u.s**2)
        # Lz = stars_orbit.angular_momentum()[2,0].to(u.km*u.kpc/u.s)
        return rguide, zmax, ecc, energies, Lz
    else:
        raise ValueError('Input arrays must have the same length.')


def add_kinematics(df, id_name='source_id', verbose=False):
    """
    Join catalog with orbital parameters for Gaia source IDs.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to join to the Gaia orbit catalog.
    id_name : str, optional
        Column name of Gaia source IDs in df. Default is 'source_id'.
    verbose : bool, optional
        If True, print verbose output to terminal.

    Returns
    -------
    df : pandas.DataFrame
        Input DataFrame merged with Gaia orbit parameters.
    """
    data = fits.open(
        paths.data / 'MWM' / 'dr3-rv-good-plx-MilkyWayPotential2022-joined.fits'
    )
    kinematic = data[1].data
    if verbose: print('Finished reading in data!')
    ids = pd.DataFrame(kinematic.source_id, columns=['source_id'])
    checklist = ids['source_id'].isin(df[id_name])
    if verbose: print('Finished matching source id, total %d stars'%(sum(checklist)))
    kinematic_dr3 = kinematic[checklist]
    
    # DataFrame with kinematic data
    kinematic_dr3 = pd.DataFrame(
        np.array((
            kinematic_dr3.source_id,
            kinematic_dr3.xyz[:,0],
            kinematic_dr3.xyz[:,1],
            kinematic_dr3.xyz[:,2],
            kinematic_dr3.vxyz[:,0],
            kinematic_dr3.vxyz[:,1],
            kinematic_dr3.vxyz[:,2],
            kinematic_dr3.actions[:,0],
            kinematic_dr3.actions[:,1],
            kinematic_dr3.actions[:,2],
            kinematic_dr3.E,
            kinematic_dr3.L[:,0],
            kinematic_dr3.L[:,1],
            kinematic_dr3.L[:,2],
            kinematic_dr3.ecc,
            kinematic_dr3.parallax,
            kinematic_dr3.ra,
            kinematic_dr3.dec,
            kinematic_dr3.phot_g_mean_mag,
            kinematic_dr3.phot_bp_mean_mag,
            kinematic_dr3.phot_rp_mean_mag,
            kinematic_dr3.ruwe,
            kinematic_dr3.z_max,
            kinematic_dr3.r_apo/2+kinematic_dr3.r_per/2
        ), dtype=str).T,
        columns=[
            'source_id','x','y','z',
            'vx','vy','vz','Jx','Jy',
            'Jz','E','Lx','Ly','Lz','e',
            'parallax','ra','dec','phot_g_mean_mag',
            'phot_bp_mean_mag','phot_rp_mean_mag',
            'ruwe','z_max','Rg']
    )
    
    for i in kinematic_dr3.columns:
        if i=='source_id':
            kinematic_dr3[i] = [int(j) for j in kinematic_dr3[i]]
            continue
        kinematic_dr3[i] = [float(j) for j in kinematic_dr3[i]]
        
    df = pd.merge(
        df, kinematic_dr3,
        left_on=id_name, right_on='source_id', how='left'
    )
    
    return df
    

if __name__ == '__main__':
    main()
