"""
Generate the Milky Way Mapper sample with quality cuts.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interpn
from astropy.io import fits
import astropy.units as u
import astropy.coordinates as coord
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from galpy.actionAngle import estimateDeltaStaeckel
import gala.dynamics as gd
import gala.potential as gp

from utils import fits_to_pandas, get_bin_centers
import paths

LOGG_CUT = (1.0, 3.5)
TEFF_CUT = (4000, 5500)
ABUND_ERR_CUT = 0.2

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
    # Fix datatype for contamination & confusion flag
    # mwm_full['cc_flg'] = mwm_full['cc_flg'].astype('str')
    # print(mwm_full['cc_flg'])
    # Quality cuts
    print('Implementing quality cuts...')
    sample = mwm_full[
        (mwm_full['sdss_id'] > 0) &
        # ASPCAP flags
        (mwm_full['sdss4_apogee_extra_target_flags'] == 0) &
        (mwm_full['flag_bad'] == 0) & 
        (mwm_full['spectrum_flags'] == 0) &
        (mwm_full['snr'] > 100) &
        (mwm_full['m_h_atm'] > -1.5) & # Meszaros et al. (2025) recommendation
        # RGB
        (mwm_full['logg'] > LOGG_CUT[0]) & 
        (mwm_full['logg'] < LOGG_CUT[1]) &
        (mwm_full['teff'] > TEFF_CUT[0]) & 
        (mwm_full['teff'] < TEFF_CUT[1]) &
        # drop stars with no abundance values
        (mwm_full['ce_h'] > -999) & 
        (mwm_full['mg_h'] > -999) &
        (mwm_full['fe_h'] > -999) &
        # Remove stars with abundance flags
        (mwm_full['ce_h_flags'] == 0) &
        (mwm_full['mg_h_flags'] == 0) &
        (mwm_full['fe_h_flags'] == 0) &
        # Limit to stars with low abundance uncertainties
        (mwm_full['e_ce_h'] < ABUND_ERR_CUT) &
        (mwm_full['e_mg_h'] < ABUND_ERR_CUT) &
        (mwm_full['e_fe_h'] < ABUND_ERR_CUT)
    ].copy()
    # drop duplicate SDSS-V IDs with the lowest SNR
    sample.sort_values(['sdss_id', 'snr'], inplace=True, ascending=True)
    sample.drop_duplicates(subset='sdss_id', keep='last', inplace=True)
    # Calculate upper limits and flag abundances below these limits
    print('Computing abundance limits...')
    sample = compute_upper_limits(sample)
    # Calculate abundance ratios and errors in quadrature
    print('Calculating abundance ratios and coordinates...')
    sample['mg_fe'], sample['e_mg_fe'] = abundance_ratio(sample, 'mg', 'fe')
    sample['ce_mg'], sample['e_ce_mg'] = abundance_ratio(sample, 'ce', 'mg')
    sample['ce_fe'], sample['e_ce_fe'] = abundance_ratio(sample, 'ce', 'fe')
    sample['c_n'], sample['e_c_n'] = abundance_ratio(sample, 'c', 'n')
    # Require Gaia distances
    sample.dropna(axis=0, how='any', subset=['r_med_photogeo'], inplace=True)
    print('Joining with orbit parameters...')
    sample = add_kinematics(sample, id_name='gaia_dr3_source_id', verbose=True)
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
    # Apply log(g) calibrations
    print('Applying log(g) calibrations...')
    sample = logg_calibrations(sample)

    # Export catalogs
    print('Exporting high-quality RGB sample (sample.csv)...')
    sample.to_csv(paths.data / 'MWM' / 'sample.csv', index=False)
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
    fitspath = paths.data / 'MWM' / 'dr3-rv-good-plx-MilkyWayPotential2022-joined.fits'
    with fits.open(fitspath) as hdul:
        kinematic = hdul[1].data
    if verbose: print('Finished reading in data!')
    ids = pd.DataFrame(kinematic.source_id, columns=['source_id'])
    checklist = ids['source_id'].isin(df[id_name])
    if verbose: print('Finished matching source id, total %d stars'%(sum(checklist)))
    kinematic_dr3 = kinematic[checklist]

    # Cartesian to cylindrical coordinates
    galcen = coord.Galactocentric(
        x=kinematic_dr3.xyz[:,0] * u.kpc, 
        y=kinematic_dr3.xyz[:,1] * u.kpc,
        z=kinematic_dr3.xyz[:,2] * u.kpc,
        v_x=kinematic_dr3.vxyz[:,0] * u.km/u.s, 
        v_y=kinematic_dr3.vxyz[:,1] * u.km/u.s, 
        v_z=kinematic_dr3.vxyz[:,2] * u.km/u.s, 
        representation_type='cartesian', 
        differential_type='cartesian',
        galcen_distance=8.275*u.kpc,
        z_sun=20.8*u.pc,
        galcen_v_sun=np.array([8.4, 251.8, 8.4]) * u.km/u.s,
    )
    galcen.representation_type = 'cylindrical'
    
    # DataFrame with kinematic data
    kinematic_dr3 = pd.DataFrame(
        np.array((
            kinematic_dr3.source_id,
            kinematic_dr3.xyz[:,0],
            kinematic_dr3.xyz[:,1],
            kinematic_dr3.xyz[:,2],
            galcen.rho.value,
            galcen.phi.value,
            kinematic_dr3.vxyz[:,0],
            kinematic_dr3.vxyz[:,1],
            kinematic_dr3.vxyz[:,2],
            galcen.d_rho.value,
            galcen.d_phi.value * galcen.rho.value,
            kinematic_dr3.actions[:,0],
            kinematic_dr3.actions[:,1],
            kinematic_dr3.actions[:,2],
            kinematic_dr3.E,
            kinematic_dr3.L[:,0],
            kinematic_dr3.L[:,1],
            kinematic_dr3.L[:,2],
            kinematic_dr3.ecc,
            kinematic_dr3.parallax,
            # kinematic_dr3.ra,
            # kinematic_dr3.dec,
            # kinematic_dr3.phot_g_mean_mag,
            # kinematic_dr3.phot_bp_mean_mag,
            # kinematic_dr3.phot_rp_mean_mag,
            kinematic_dr3.ruwe,
            kinematic_dr3.z_max,
            kinematic_dr3.r_apo/2+kinematic_dr3.r_per/2
        ), dtype=str).T,
        columns=[
            'source_id','galx','galy','galz','galr','galphi',
            'vx','vy','vz','vr','vphi',
            'Jx','Jy','Jz','E','Lx','Ly','Lz','ecc',
            'parallax','ruwe','z_max','Rg']
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


def compute_upper_limits(df):
    """
    Compute Shetrone et al. (2025) upper limits for abundance measurements,
    and flag abundances lower than the limit.
    """
    coeffs = pd.read_csv(
        paths.data / 'MWM' / 'shetrone_dr17_limits.csv', index_col='species'
    )
    for el, row in coeffs.iterrows():
        df['lim_%s_h' % el] = abund_limit_func(
            df['teff'], df['snr'], **row.to_dict()
        )
        df['lim_%s_h_flag' % el] = (
            df['%s_h' % el] - df['e_%s_h' % el] <= df['lim_%s_h' % el]
        ).astype(np.int64)
    return df


def abund_limit_func(teff, snr, alpha=0, beta=1, gamma=1, delta=1):
    """
    Generic upper limit function based on Teff and S/N from Shetrone et al. (2025).
    """
    return alpha + beta*(teff/1000) + gamma*(np.log10(snr)) + delta*(np.log10(snr))**2


def logg_calibrations(df):
    """
    Apply calibrations to correct for abundance correlations with log(g).
    """
    # Initialize grid of log(g), [Mg/H] values
    MgH_bin_edges = np.round(np.linspace(-0.75, 0.45, 13, endpoint=True), 2)
    MgH_bin_centers = get_bin_centers(MgH_bin_edges)
    logg_bin_edges = np.linspace(0, 3.5, 8, endpoint=True)
    logg_bin_centers = get_bin_centers(logg_bin_edges)
    grid = (MgH_bin_centers, logg_bin_centers)
    # Load calibration grids
    fe_offsets = np.load(paths.data / 'MWM' / 'fe_offset_grid.npy')
    ce_offsets = np.load(paths.data / 'MWM' / 'ce_offset_grid.npy')

    # Interpolate & apply log(g) corrections
    feh_corr = np.empty(df.shape[0])
    ceh_corr = np.empty(df.shape[0])
    for i in range(df.shape[0]):
        feh_corr[i] = apply_elem_offsets(
            df['mg_h'].iloc[i], 
            df['logg'].iloc[i], 
            df['fe_h'].iloc[i], 
            fe_offsets,
            grid
        )
        ceh_corr[i] = apply_elem_offsets(
            df['mg_h'].iloc[i], 
            df['logg'].iloc[i], 
            df['ce_h'].iloc[i], 
            ce_offsets,
            grid
        )

    # Apply to sample DataFrame
    df['fe_h_corr'] = feh_corr
    df['ce_h_corr'] = ceh_corr
    df['fe_mg_corr'] = df['fe_h_corr'] - df['mg_h']
    df['mg_fe_corr'] = df['mg_h'] - df['fe_h_corr']
    df['ce_mg_corr'] = df['ce_h_corr'] - df['mg_h']
    df['ce_fe_corr'] = df['ce_h_corr'] - df['fe_h_corr']
    return df


def apply_elem_offsets(mgh, logg, xh, offsets, grid):
    interp_point = np.array([mgh, logg])
    star_offset = interpn(grid, offsets, interp_point, bounds_error=False, fill_value=None)[0]
    corr_xh = xh + star_offset
    return corr_xh
    

if __name__ == '__main__':
    main()
