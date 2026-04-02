"""
Plot the local [Ce/Mg]--age relation and fit a linear trend in bins
of metallicity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator
from scipy import stats
from scipy.optimize import minimize
import emcee
import corner
from odrpack import odr_fit

from plotting import ONE_COLUMN_WIDTH
from utils import sample_rows
import paths

MET_COL = 'fe_h_corr' # Column with metallicity values
MET_LABEL = r'[Fe/H]'
AGE_FIT_RANGE = (1, 8) # Range of ages to fit linear trend
RLIM = (7, 9)
ZLIM = (0, 0.5)
ABUND_SCALE = 1 # scale abundance values for MCMC fitting


def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')

    # Data
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    local_sample = mwm_rgb[
        (mwm_rgb['Rg'] >= RLIM[0]) &
        (mwm_rgb['Rg'] < RLIM[1]) &
        (mwm_rgb['z_max'] >= ZLIM[0]) &
        (mwm_rgb['z_max'] < ZLIM[1]) &
        (mwm_rgb['good_age'])
    ].copy()
    # Restrict age trends to low-alpha stars only
    local_low_alpha = local_sample[local_sample['low_alpha']]
    local_high_alpha = local_sample[~local_sample['low_alpha']] # include border stars

    # Metallicity bins
    met_bins = np.arange(-0.5, 0.51, 0.2)
    age_arr = np.arange(0, 12.1, 0.1)

    nrows = len(met_bins)-1
    fig, axs = plt.subplots(
        nrows, 1,
        figsize=(ONE_COLUMN_WIDTH, 0.6*nrows*ONE_COLUMN_WIDTH),
        sharex=True, sharey=True, gridspec_kw={'hspace': 0.},
    )
    cmap = plt.get_cmap('viridis')
    norm = BoundaryNorm(met_bins, cmap.N, extend='both')
    xlim = (0, 11)
    ylim = (-0.7, 0.9)

    fits = []
    for i, ax in enumerate(axs):
        # Underlying scatter plot of all low-alpha stars
        pcm = axs[i].hexbin(
            local_sample['age'], local_sample['ce_mg_corr'],
            C=np.ones(local_sample.shape[0]),
            reduce_C_function=np.sum,
            gridsize=(30, 12),
            cmap='binary',
            linewidths=0.2,
            mincnt=1,
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]]
        )
        # ax.scatter(
        #     local_low_alpha['age'], local_low_alpha['ce_mg_corr'],
        #     c='gray', s=1, marker='o', rasterized=True, edgecolor='none'
        # )
        # Plot high-alpha stars for reference (not fit)
        # ax.scatter(
        #     local_high_alpha['age'], local_high_alpha['ce_mg_corr'], 
        #     edgecolors='gray', 
        #     s=1, marker='o', rasterized=True, facecolors='w', linewidths=0.3
        # )
        # Bin by metallicity and fit linear trend to stars within good age range
        met_lim = (np.round(met_bins[-(i+2)], 2), np.round(met_bins[-(i+1)], 2))
        met_center = np.mean(met_lim) # mean metallicity of bin
        color = cmap(norm(met_center))
        # Scatter plot of stars in metallicity range
        subset_low_alpha = local_sample[
            (local_sample[MET_COL] >= met_lim[0]) &
            (local_sample[MET_COL] < met_lim[1]) &
            (local_sample['low_alpha'])
        ]
        ax.scatter(
            subset_low_alpha['age'], subset_low_alpha['ce_mg_corr'],
            color=color, s=3, marker='o', rasterized=True, edgecolor='none'
        )
        # Plot high-alpha stars for reference (not fit)
        subset_high_alpha = local_sample[
            (local_sample[MET_COL] >= met_lim[0]) &
            (local_sample[MET_COL] < met_lim[1]) &
            (~local_sample['low_alpha'])
        ]
        ax.scatter(
            subset_high_alpha['age'], subset_high_alpha['ce_mg_corr'], 
            edgecolors=color, 
            s=3, marker='o', rasterized=True, facecolors='w', linewidths=0.5
        )
        # Casali et al. (2025) relation for comparison
        ax.plot(age_arr, casali_relation(age_arr, met_center), 'w-', lw=2)
        ax.plot(
            age_arr, casali_relation(age_arr, met_center), 'k:', 
            label='Casali et al. (2025)'
        )
        # Fit linear age trend
        subset_fit = local_sample[
            (local_sample[MET_COL] >= met_lim[0]) & 
            (local_sample[MET_COL] < met_lim[1]) &
            (local_sample['age'] >= AGE_FIT_RANGE[0]) &
            (local_sample['age'] < AGE_FIT_RANGE[1]) &
            (local_sample['low_alpha'])
        ]
        regress = stats.linregress(subset_fit['age'], subset_fit['ce_mg_corr'])
        fits.append(regress)
        # Plot linear regression
        yfit = age_arr * regress.slope + regress.intercept
        # White outline for plot legibility
        ax.plot(age_arr, yfit, linestyle='-', linewidth=2, color='w')
        ax.plot( # extends beyond fit region
            age_arr[age_arr < AGE_FIT_RANGE[0]], 
            yfit[age_arr < AGE_FIT_RANGE[0]], 
            linestyle='--', 
            color=color
        )
        ax.plot( # segment within fit region
            age_arr[(AGE_FIT_RANGE[0] <= age_arr) & (age_arr < AGE_FIT_RANGE[1])], 
            yfit[(AGE_FIT_RANGE[0] <= age_arr) & (age_arr < AGE_FIT_RANGE[1])], 
            linestyle='-', 
            color=color
        )
        ax.plot( # extends beyond fit region
            age_arr[age_arr >= AGE_FIT_RANGE[1]], 
            yfit[age_arr >= AGE_FIT_RANGE[1]], 
            linestyle='--', 
            color=color
        )
        ax.set_title(
            r'$%s\leq$%s$<%s$' % (met_lim[0], MET_LABEL, met_lim[1]),
            y=0.95, pad=0, va='top',
            bbox={
                'facecolor': 'w', 
                'edgecolor': 'none', 
                'alpha': 1.,
                'pad': 0.2,
                'boxstyle': 'round'
            }
        )
        # Plot MCMC
        print(met_lim[0], met_lim[1])
        # flat_samples = mcmc_fit(subset_fit, plot=True, verbose=True) / ABUND_SCALE
        # inds = np.random.randint(len(flat_samples), size=100)
        # for ind in inds:
        #     sample = flat_samples[ind]
        #     ax.plot(age_arr, sample[0] * age_arr + sample[1], color=color, alpha=0.1)
        # med_fit = np.median(flat_samples, axis=0)
        # print(med_fit)
        # ax.plot(age_arr, med_fit[0] * age_arr + med_fit[1], 'k-')
        # ODR
        fit_params, fit_sd = get_odr_fit(subset_low_alpha, verbose=True)
        ax.plot(age_arr, fit_params[0] * age_arr + fit_params[1], 'k-')


    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)
    axs[0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[-1].set_xlabel('Age [Gyr]')
    for ax in axs:
        ax.set_ylabel(r'[Ce/Mg]$_{\rm corr}$')

    plt.savefig(paths.figures / 'metallicity_fits')


def model(x, beta):
    m, b = beta
    return m * x + b


def log_prior(theta):
    m, b = theta[:2]
    # x = theta[2:]
    if -20 < m < 20 and -20 < b < 20:# and all((x > 0) & (x < 20)):
        return 0
    else:
        return -np.inf


def log_likelihood(theta, xobs, yobs, xerr, yerr):
    m, b = theta[:2]
    # x = theta[2:]
    y = m * xobs + b
    sigma2 = yerr**2 + m**2 * xerr**2
    return -0.5 * np.sum((yobs - y)**2 / sigma2 + np.log(sigma2))
    # return -0.5 * (np.sum(((yobs - y) / yerr) ** 2) +
    #                np.sum(((xobs - x) / xerr) ** 2))


def log_probability(theta, xobs, yobs, xerr, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, xobs, yobs, xerr, yerr)


def get_odr_fit(data, verbose=False):
    # Parse data
    xobs = data['age'].to_numpy()
    yobs = data['ce_mg_corr'].to_numpy()
    xerr = 0.5 * ((data['e_p_age'] - data['age']) + 
                  (data['age'] - data['e_n_age'])).to_numpy()
    yerr = data['e_ce_mg'].to_numpy()
    beta0 = [0, 0] # initial guess
    lower, upper = [-5, -5], [5, 5] # bounds
    if verbose:
        report = 'short'
    else:
        report = 'none'
    sol = odr_fit(
        model, xobs, yobs, beta0, 
        bounds = (lower, upper),
        weight_x = 1 / xerr, weight_y = 1 / yerr,
        report = report
    )
    return sol.beta, sol.sd_beta


def mcmc_fit(data, nwalkers=32, max_steps=5000, burnin=100, thin=20, verbose=False, seed=None, plot=False):
    rng = np.random.default_rng(seed)
    # Parse data
    # data.sort_values('age', inplace=True)
    xobs = data['age'].to_numpy()
    yobs = data['ce_mg_corr'].to_numpy() * ABUND_SCALE
    xerr = 0.5 * ((data['e_p_age'] - data['age']) + 
                  (data['age'] - data['e_n_age'])).to_numpy()
    yerr = data['e_ce_mg'].to_numpy() * ABUND_SCALE
    # Max likelihood solution
    nll = lambda *args: -log_likelihood(*args)
    nparams = 2
    ndim = nparams# + data.shape[0]
    # nwalkers = ndim * 2
    # print('nwalkers = %s' % nwalkers)
    initial = ABUND_SCALE * np.array([-0.05, 0.5]) + 1e-2 * rng.standard_normal(ndim)
    ml_soln = minimize(nll, initial, args=(xobs, yobs, xerr, yerr))
    print(ml_soln.x)
    # Set up sampler
    theta0 = ml_soln.x + 1e-2 * rng.standard_normal((nwalkers, ndim))
    # initial = np.zeros(nparams)
    # theta0 = [np.append(initial, xobs) + 1e-3 * rng.standard_normal(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=[xobs, yobs, xerr, yerr]
    )
    # Sample
    if verbose: print('Sampling...')
    sampler.run_mcmc(theta0, max_steps, progress=verbose)
    # print('Calculating autocorrelation time...')
    # Flatten and remove burnin
    tau = sampler.get_autocorr_time(quiet=True)
    burnin = int(2 * np.max(tau))
    print('burnin = %s' % burnin)
    thin = int(0.5 * np.min(tau))
    flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    print(flat_samples.shape)
    print(np.median(flat_samples, axis=0))
    flat_samples = flat_samples[:,:nparams]
    # Diagnostic plots
    if plot:
        if verbose: print('Plotting chains...')
        plot_dir = paths.extra / 'mcmc'
        if not plot_dir.is_dir():
            plot_dir.mkdir(parents=True)
        # Plot chains
        labels = ['m', 'b']
        full_samples = sampler.get_chain()
        fig, axs = plt.subplots(ndim, figsize=(4, nparams), sharex=True)
        for i in range(nparams):
            axs[i].plot(full_samples[:,:100,i], 'k', alpha=0.1)
            axs[i].set_ylabel(labels[i])
        axs[-1].set_xlabel('Step number')
        plt.savefig(plot_dir / 'chains.png')
        plt.close()
        # Cornerplot
        print(flat_samples.shape)
        fig = corner.corner(flat_samples, labels=labels)
        plt.savefig(plot_dir / 'cornerplot.png')
        plt.close()
    if verbose: print('Done!')
    del sampler
    del full_samples
    return flat_samples



def casali_relation(age, feh):
    """
    Global fit to the age-[Ce/Mg] relation from Casali et al. (2025).
    """
    return -0.032 * age + 0.194 * feh + 0.092


if __name__ == '__main__':
    main()
