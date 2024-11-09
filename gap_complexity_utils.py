"""
Functions and data structures for finding the gap complexity of a secularly
evolving planetary system.
Joseph Livesey, 2024
"""


import numpy as np
import astropy.units as u
import astropy.constants as const
import pandas as pd
import itertools
import pickle
from scipy.special import lambertw
# from copy import deepcopy
# from ctypes import cdll, c_double

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn

import rebound
import celmech
from celmech.secular import LaplaceLagrangeSystem
from celmech.nbody_simulation_utilities \
import reb_orbits, get_simarchive_integration_results

import warnings
warnings.filterwarnings('ignore', category=np.exceptions.ComplexWarning)


to_radians = np.pi/180.0
to_degrees = 180.0/np.pi


class NewError(Exception):
    pass


class UndefError(Exception):

    def __init__(self, attr, func):
        self.attr = attr
        self.func = func
        self.message = '{attr} is not set. First run {func}.'.format(
            attr=self.attr, func=self.func)

    def __str__(self):
        return self.message


class EnsemblePair:
    """
    An ensemble of gap complexity simulation pairs, with each pair containing a
    system with and a system without an outer giant companion.

     Attributes
    ----------
    num_simulations : int
        Number of simulation pairs in the ensemble.
    stip_mult : int
        Number of planets in the STIP.
    og_mult : int
        Number of outer giant companions.
    inc_scale : float
        Scale of the Rayleigh distribution from which inclinations are sampled.
    og_inc : float
        Orbital inclination of the companion.
    rng_seed : int
        Seed for random number generation.
    rng : numpy.random.Generator
        Random number generator with a prescribed seed.
    vlim : float
        Colorbar limits in heatmaps are +/- vlim.
    save_simulation_pairs : bool
        Keep all data from each simulation pair? If True, might run out of
        memory while running the ensemble.
    randomize_inclinations : bool
        Draw new STIP inclinations for each SimulationPair in the ensemble?
    init : bool
        Has the sample of hypothetical systems been constructed?
    done : bool
        Have all Laplace--Lagrange solutions for the systems been computed?
    og_mass_dist : function
        Function that generates random masses for the companion.
    og_sma_dist : function
        Function that generates random semi-major axes for the companion.
    method : str
        Sampling method: either `random` or `grid`.
    pairs : numpy.ndarray
        Array of simulation pairs.
    gc_with : numpy.ndarray
        Array of time-averaged gap complexity for all systems with a companion.
    gc_wout : numpy.ndarray
        Array of time-averaged gap complexity for all systems without a
        companion.
    gc_err : numpy.ndarray
        Array of changes in average gap complexity incurred by the companion
        for all simulation pairs.
    proximity_to_resonance : numpy.ndarray
        Array of proximities to secular resonances for all systems with a
        companion.
    mass_array : numpy.ndarray
        All sampled companion masses.
    sma_array : numpy.ndarray
        All sampled companion semi-major axes.
    alpha_array : numpy.ndarray
        Ratio between companion semi-major axis and outermost STIP planet semi-
        major axis for all sampled companion semi-major axes.
    
    Methods
    -------
    _rng_func(dist, *args)
        Wraps an arbitrary statistical distribution from numpy.random.
    set_og_mass_dist(dist, *args)
        Defines the distribution over which we sample masses for the companion.
    set_og_sma_dist(dist, *args)
        Defines the distribution over which we sample semi-major axes for the
        companion.
    _random_orbit_separation(sma_first, sma_max)
        Gets random orbital distances for an additional companion that comply
        with system-dependent rules.
    sample(method='random', mass_min=1.0e-3, mass_max=0.5, sma_min=0.75,
           sma_max=10.0))
        Generates a sample of hypothetical systems, either randomly or from a
        grid.
    run()
        Computes the Laplace--Lagrange solutions and resulting gap complexity
        evolutions for every system in the sample.
    get_proximity_to_resonance()
        Calculates the proximity of the system to a first-, second-, third-, or
        fourth-order resonance between inclination eigenfrequencies.
    histogram(save=False)
        Generates a histogram of time-averaged gap complexities for all
        simulations in a random sample.
    heatmaps(save=False, output=False)
        Generates a heatmap in the change in gap complexity induced by the
        companion.
    secular_resonance_heatmap(save=False, output=False)
        Generates a map of secular resonances in the parameter space being
        explored.
    to_dataframe(save=False, filename=None)
        Saves the essential information from an ensemble in a pandas dataframe.
    save(filename=None)
        Saves an ensemble in a pkl file.
    load(filename)
        Re-creates an ensemble from a pkl file.
    """

    def __init__(self, num_simulations=100, stip_mult=4, og_mult=1,
                 inc_scale=2.5, og_inc=10.0, rng_seed=None, vlim=0.15,
                 save_simulation_pairs=True, randomize_inclinations=True):
        # Class attributes
        self.num_simulations = num_simulations
        self.stip_mult = stip_mult
        self.og_mult   = og_mult
        self.inc_scale = inc_scale
        self.og_inc    = og_inc
        if rng_seed is not None:
            self.rng_seed = rng_seed
            self.rng = np.random.default_rng(rng_seed)
        else:
            pass
        self.vlim = vlim
        self.save_simulation_pairs = save_simulation_pairs
        self.randomize_inclinations = randomize_inclinations
        self.init = False
        self.done = False

        # Plot params
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['font.family'] = 'cmu-serif'
        mpl.rcParams['font.size'] = 16
        return

    def _rng_func(self, dist, *args):
        """
        Wraps an arbitrary statistical distribution from numpy.random.

        Parameters
        ----------
        dist : function
            The type of statistical distribution.
        
        Returns
        -------
        float
            A random number sampled from that distribution.
        """
        def rn():
            return dist(*args)
        return rn

    def set_og_mass_dist(self, dist, *args):
        """
        Defines the distribution over which we sample masses for the companion.

        Parameters
        ----------
        dist : function
            The type of statistical distribution.
        """
        self.og_mass_dist = self._rng_func(dist, *args)
        return

    def set_og_sma_dist(self, dist, *args):
        """
        Defines the distribution over which we sample semi-major axes for the
        companion.

        Parameters
        ----------
        dist : function
            The type of statistical distribution.
        """
        self.og_sma_dist = self._rng_func(dist, *args)
        return

    def _random_orbit_separation(self, sma_first, sma_max):
        """
        Gets random orbital distances for an additional companion that comply
        with system-dependent rules.

        Parameters
        ----------
        sma_first : float
            The semi-major axis of the previously added companion.
        sma_max : float
            Maximum semi-major axis allowed.
        
        Returns
        -------
        float
            A semi-major axis for the new companion.
        """
        stdev = 0.1
        # Set mean new semimajor axis at location of 2:1 MMR (idk)
        loc = (4.0**(1.0/3) - 1.0) * sma_first
        delta_a = self.rng.normal(loc, stdev)
        if delta_a <= 0.0:
            return self._random_orbit_separation(sma_first, sma_max)
        else:
            return delta_a

    def sample(self, method='random', mass_min=1.0e-3, mass_max=0.5,
               sma_min=0.75, sma_max=10.0, ascale='linear'):
        """
        Generates a sample of hypothetical systems, either randomly or from a
        grid.

        Parameters
        ----------
        method : str
            The sampling method: either `random` or `grid`.
        mass_min : float
            Minimum mass for the companion.
        mass_max : float
            Maximum mass for the companion.
        sma_min : float
            Minimum semi-major axis for the companion.
        sma_max : float
            Maximum semi-major axis for the companion.
        """
        self.method = method
        self.ascale = ascale
        if method == 'random':
            try:
                foo = self.og_mass_dist
                bar = self.og_sma_dist
            except AttributeError as err:
                if err.name == 'og_mass_dist':
                    raise UndefError('OG mass',
                                     'EnsemblePair.set_og_mass_dist')
                elif err.name == 'og_sma_dist':
                    raise UndefError('OG semimajor axis',
                                     'EnsemblePair.set_og_sma_dist')
                else:
                    raise err
            self.pairs = np.empty((self.num_simulations), dtype=SimulationPair)
            mass_list = []
            sma_list  = []
            for _ in range(self.num_simulations):
                for _ in range(self.og_mult):
                    m = self.og_mass_dist()
                    mass_list.append(m)
                a = self.og_sma_dist()
                sma_list.append(a)
            for i in range(self.num_simulations):
                sim_pair = SimulationPair(self.stip_mult, self.inc_scale,
                                          self.og_inc, stellar_radius=0.005,
                                          rng_seed=self.rng_seed+i)
                aj = sma_list[i]
                for j in range(self.og_mult):
                    sim_pair.add(m=mass_list[i+j], a=aj)
                    aj += self._random_orbit_separation(aj, sma_max)
                self.pairs[i] = sim_pair # only need 1D for the random sampling
            self.mass_array = np.array(mass_list)
            self.sma_array  = np.array(sma_list)
        elif method == 'grid':
            side_len = np.sqrt(self.num_simulations)
            if side_len % 1 != 0:
                raise NewError('`num_simulations` must be a perfect square for \
                                grid sampling!')
            side_len = int(side_len)
            self.pairs = np.empty((side_len, side_len), dtype=SimulationPair)
            # self.mass_array = np.linspace(mass_min, mass_max, side_len)
            self.mass_array = np.logspace(np.log10(mass_min),
                                          np.log10(mass_max), side_len)
            if ascale == 'linear':
                self.sma_array = np.linspace(sma_min, sma_max, side_len)
            elif ascale == 'log':
                self.sma_array = np.logspace(np.log10(sma_min),
                                             np.log10(sma_max), side_len)
            elif ascale == 'powertower':
                arr = np.linspace(unspindle(sma_min), unspindle(sma_max),
                                  side_len)
                self.sma_array = arr ** arr
            else:
                raise NewError('ERROR: Unrecognized option for `ascale`.') 
            dseed = 0
            for i in range(side_len):
                for j in range(side_len):
                    if self.randomize_inclinations:
                        dseed = i * side_len + j # Amount by which to advance seed
                    sim_pair = SimulationPair(self.stip_mult, self.inc_scale,
                                              self.og_inc, stellar_radius=0.005,
                                              rng_seed=self.rng_seed+dseed)
                    sim_pair.add(m=self.mass_array[i], a=self.sma_array[j])
                    self.pairs[i][j] = sim_pair # need 2D for grid sampling
        else:
            raise NewError('Sampling method must be `grid` or `random`.')
        self.init = True
        return

    def run(self):
        """
        Computes the Laplace--Lagrange solutions and resulting gap complexity
        evolutions for every system in the sample.
        """
        if not self.init:
            self.sample()
        self.gc_with = np.empty(self.pairs.shape)
        self.gc_wout = np.empty(self.pairs.shape)
        self.hill_stable = np.empty(self.pairs.shape)
        self.amd_metric = np.empty(self.pairs.shape)
        if self.method == 'random':
            for sim_pair in self.pairs:
                sim_pair.get_ll_systems()
                sim_pair.get_ll_solutions()
                sim_pair.get_gap_complexities()
            for i in range(self.num_simulations):
                self.gc_with[i] = self.pairs[i].mean_gc_with
                self.gc_wout[i] = self.pairs[i].mean_gc_wout
                self.amd_metric[i] = self.pairs[i].amd_metric()
                if not self.save_simulation_pairs:
                    self.pairs[i] = None
        elif self.method == 'grid':
            for i, j in itertools.product(
                range(int(np.sqrt(self.num_simulations))),
                range(int(np.sqrt(self.num_simulations)))
            ):
                self.pairs[i][j].get_ll_systems()
                self.pairs[i][j].get_ll_solutions()
                self.pairs[i][j].get_gap_complexities()
                self.gc_with[i][j] = self.pairs[i][j].mean_gc_with
                self.gc_wout[i][j] = self.pairs[i][j].mean_gc_wout
                self.hill_stable[i][j] = \
                    self.within_stability_limit(self.pairs[i][j])
                self.amd_metric[i][j] = self.pairs[i][j].amd_metric()
                if not self.save_simulation_pairs:
                    if i != 0 or j != 0:
                        self.pairs[i][j] = None
                    else:
                        pass
        else:
            raise NewError('Sampling method must be `grid` or `random`.')
        self.done = True
        return
    
    def get_proximity_to_resonance(self):
        """
        Calculates the proximity of the system to a first-, second-, third-, or
        fourth-order resonance between inclination eigenfrequencies.
        """
        if not self.init:
            self.sample()
        if self.method != 'grid':
            raise NewError('Resonance sweep only works with `grid` sampling \
                            method.')
        else:
            self.proximity_to_resonance = np.empty(self.pairs.shape)
            for i, j in itertools.product(
                range(int(np.sqrt(self.num_simulations))),
                range(int(np.sqrt(self.num_simulations)))
            ):
                self.pairs[i][j].get_ll_systems()
                freqs = self.pairs[i][j].sys_with.inclination_eigenvalues()
                freqs.sort()
                self.proximity_to_resonance[i][j] = np.abs(
                    degree_of_commensurability(freqs)
                )
        return
    
    def within_stability_limit(self, sim_pair,
                               num_hill_radii=2.0*np.sqrt(3.0)):
        """
        Does the configuration including the OG satisfy a user-specified
        stability criterion? This criterion is provided in terms of the
        minimum number of mutual Hill radii that must separate the closest pair
        of adjacent planets.

        Returns
        -------
        bool
            Does the system satisfy this condition?
        """
        sim = sim_pair.with_og
        separation = sim.particles[-1].a - sim.particles[-2].a
        mhr = mutual_hill_radius_particles(
            sim.particles[0], sim.particles[-2], sim.particles[-1]
        )
        if separation < num_hill_radii * mhr:
            return True
        else:
            return False
    
    def _mask_nonsecular(self):
        return self.hill_stable
    
    # def _mask_nonsecular(self, num_hill_radii=2.0*np.sqrt(3.0)):
    #     """
    #     Mask out the region of parameter space in which the OG is within so
    #     many mutual Hill radii of the outermost STIP planet.

    #     Parameters
    #     ----------
    #     num_hill_radii : float
    #         The OG must be at least this many mutual Hill radii away to accept
    #         the secular solution.
        
    #     Returns
    #     -------
    #     numpy.ndarray (bool)
    #         The mask.
    #     """
    #     mask = np.zeros(self.pairs.shape, dtype=bool)
    #     for i, j in itertools.product(
    #         range(int(np.sqrt(self.num_simulations))),
    #         range(int(np.sqrt(self.num_simulations)))
    #     ):
    #         sim = self.pairs[i][j].with_og
    #         separation = sim.particles[-1].a - sim.particles[-2].a
    #         mhr = mutual_hill_radius_particles(
    #             sim.particles[0], sim.particles[-2], sim.particles[-1]
    #         )
    #         if separation < num_hill_radii * mhr:
    #             mask[i][j] = True
    #         else:
    #             pass
    #     return mask
    
    def _draw_stability_boundary(self, ax, num_hill_radii=2.0*np.sqrt(3.0)):
        """
        Draws a curve showing at what mass the OG is within so many mutual Hill
        radii of the outermost STIP planet.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The subplot on which to draw the curve.
        num_hill_radii : float
            The OG must be at least this many mutual Hill radii away to accept
            the secular solution.
        """
        cmr = np.vectorize(critical_mass_ratio)
        max_mass = cmr(
            self.pairs[0][0].with_og.particles[-2].m,
            self.pairs[0][0].with_og.particles[-2].a,
            self.sma_array,
            num_hill_radii
        )
        ax.plot(self.alpha_array, max_mass, c='k', ls='dashed', lw=5)
        return

    def histogram(self, save=False):
        """
        Generates a histogram of time-averaged gap complexities for all
        simulations in a random sample.

        Parameters
        ----------
        save : bool
            Save as a PDF?
        """
        if not self.done:
            self.run()
        else:
            pass
        gc_with = None
        gc_wout = None
        if self.method == 'random':
            gc_with = self.gc_with
            gc_wout = self.gc_wout
        elif self.method == 'grid':
            gc_with = self.gc_with.ravel()
            gc_wout = self.gc_wout.ravel()
        fig, axes = plt.subplots(2, 1, dpi=200)
        cmax = print_cmax(tabular=False)[self.stip_mult-1]
        axes[0].hist(gc_with, bins=[k * cmax/10 for k in range(10)],
                     weights=np.ones_like(gc_with)/self.num_simulations)
        axes[1].hist(gc_wout, bins=[k * cmax/10 for k in range(10)],
                     weights=np.ones_like(gc_wout)/self.num_simulations)
        # axes[0].hist(gc_with, bins=int(self.num_simulations/10),
        #              range=(0.0, cmax))
        # axes[1].hist(gc_wout, bins=int(self.num_simulations/10),
        #              range=(0.0, cmax))
        for ax in axes:
            ax.set_xlabel(r'$\langle \tilde{\mathcal{C}} \rangle$')
            ax.set_ylabel('Fraction of simulations')
        axes[0].set_title('With OG')
        axes[1].set_title('Without OG')
        fig.tight_layout()
        if save:
            fig.savefig('new-histogram.pdf')
        else:
            pass
        return

    def heatmaps(self, save=False, mask=False, draw_boundary=False,
                 cmap=mpl.cm.Spectral, output=False):
        """
        Generates a heatmap in the change in gap complexity induced by the
        companion.

        Parameters
        ----------
        save : bool
            Save as a PDF?
        mask : bool
            Mask out likely non-secular scenarios?
        draw_boundary : bool
            Draw the curve of maximum stable OG masses?
        output : bool
            Return the subplots?
        
        Returns
        -------
        tuple (matplotlib.figure.Figure, matplotlib.axes.Axes) or None
            The generated heatmaps.
        """
        if self.method != 'grid':
            raise NewError('Heatmaps can be made only for data sampled over a \
                            grid.')
        if not self.done:
            self.run()
        else:
            pass
        # fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=200, sharex=True)
        fig, ax = plt.subplots(1, 1, dpi=200)
        self.alpha_array = self.sma_array / self.pairs.ravel()[0].wout_og.particles[-1].a # ratio between OG sma and that of outermost TIP
        self.gc_err = self.gc_with - self.gc_wout
        print(np.nanmin(self.gc_err), np.nanmax(self.gc_err), np.nanmean(self.gc_err.ravel()), np.nanstd(self.gc_err.ravel()))
        print('Average change in gap complexity metric: {}\n'.format(np.nanmean(self.gc_err.ravel())))
        if self.vlim is None:
            self.vlim = np.abs(max([np.nanmin(self.gc_err), np.nanmax(self.gc_err)]))
        else:
            pass
        if mask:
            self.gc_err = np.ma.array(self.gc_err, mask=self._mask_nonsecular())
        # axes[0].contourf(self.mass_array, self.sma_array, self.gc_with, cmap=cmap)
        # axes[1].contourf(self.mass_array, self.sma_array, self.gc_wout, cmap=cmap)
        # axes[2].contourf(self.mass_array, self.sma_array, gc_err, cmap=cmap)
        # divider = make_axes_locatable(axes[0])
        # cax = divider.new_vertical(size='5%', pad=0.6, pack_start=True)
        # fig.add_axes(cax)
        # cax, _ = mpl.colorbar.make_axes(axes[2])
        _a, _m = np.meshgrid(self.alpha_array, self.mass_array)
        ax.pcolormesh(_a, _m, self.gc_err, vmin=-self.vlim, vmax=self.vlim,
                      cmap=cmap)
        cb = plt.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap,
                                #   norm=mpl.colors.Normalize(gc_err.min(),
                                #                             gc_err.max())
                                  norm=mpl.colors.Normalize(-self.vlim,
                                                            self.vlim)
            ),
            # ax=axes[2],
            ax=ax,
        )
        if self.ascale == 'log':
            ax.set_xscale('log')
        elif self.ascale == 'powertower':
            ax.set_xscale('function',
                          functions=(lambda x: x**x, lambda x: unspindle(x)))
        ax.set_yscale('log')
        # for ax in axes:
        #     ax.set_xlabel('OG mass')
        # axes[0].set_ylabel(r'$\alpha$')
        ax.set_xlabel(r'$\alpha^{-1}$')
        ax.set_ylabel(r'$m_\mathrm{OG}/m_\star$')
        # ax.invert_yaxis()
        cb.set_label(
            r'$\langle \tilde{\mathcal{C}} \rangle_2 - \langle \tilde{\mathcal{C}} \rangle_1 $')
        if draw_boundary:
            self._draw_stability_boundary(ax)
        fig.tight_layout()
        if save:
            fig.savefig('new-heatmaps.pdf')
        if output:
            return fig, ax
        else:
            return
    
    def secular_resonance_heatmap(self, save=False, output=False):
        """
        Generates a map of secular resonances in the parameter space being
        explored.

        Parameters
        ----------
        save : bool
            Save as a PDF?
        output : bool
            Return the subplots?
        
        Returns
        -------
        tuple (matplotlib.figure.Figure, matplotlib.axes.Axes) or None
            The generated heatmaps.
        """
        self.get_proximity_to_resonance()
        cmap = mpl.cm.magma_r
        fig, ax = plt.subplots(1, 1, dpi=200)
        self.alpha_array = self.sma_array / self.pairs.ravel()[0].wout_og.particles[-1].a # ratio between OG sma and that of outermost TIP
        if self.vlim is None:
            # self.vlim = max(np.abs([self.proximity_to_resonance.min(),
            #                         self.proximity_to_resonance.max()]))
            self.vlim = self.proximity_to_resonance.max()
        else:
            pass
        _a, _m = np.meshgrid(self.alpha_array, self.mass_array)
        ax.pcolormesh(_a, _m, self.proximity_to_resonance, vmin=0.0,
                      vmax=self.vlim, cmap=cmap)
        cb = plt.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap,
                                  norm=mpl.colors.Normalize(0.0,
                                                            self.vlim)
            ),
            ax=ax
        )
        ax.set_yscale('log')
        ax.set_xlabel(r'$\alpha^{-1}$')
        ax.set_ylabel(r'$m_\mathrm{OG}/m_\star$')
        cb.set_label('Proximity to resonance')
        fig.tight_layout()
        if save:
            fig.savefig('secular-resonances.pdf')
        if output:
            return fig, ax
        else:
            return

    def to_dataframe(self, save=False, filename=None):
        """
        Saves the essential information from an ensemble in a pandas dataframe.

        Parameters
        ----------
        save : bool
            Save in a CSV file?
        filename : str
            Target location for the CSV file.
        
        Returns
        -------
        pandas.DataFrame
            A dataframe containing the essential information from the ensemble.
        """
        if not self.done:
            self.run()
        if filename is None:
            filename = 'data/{}_tips_{}_ogs.csv'.\
                       format(self.stip_mult, self.og_mult)
        cmax = print_cmax(False)[self.stip_mult-1]
        og_mass = []
        og_sma  = []
        gc_with = []
        gc_wout = []
        gc_diff = []
        for pair in self.pairs.ravel():
            og_mass.append(pair.og_mass)
            og_sma.append(pair.og_sma)
            gc_with.append(np.nanmean(pair.gc_with))
            gc_wout.append(np.nanmean(pair.gc_wout))
            gc_diff.append(
                np.nanmean(pair.gc_with) - np.nanmean(pair.gc_wout)
            )
        data_dict = {
            'og_mass': og_mass,
            'og_sma': og_sma,
            'gc_with': gc_with,
            'gc_wout': gc_wout,
            'gc_diff': gc_diff,
        }
        df = pd.DataFrame.from_dict(data_dict)
        if save:
            df.to_csv(filename)
        return df

    def save(self, filename=None):
        """
        Saves an ensemble in a pkl file.

        Parameters
        ----------
        filename : str
            Target location for the pkl file.
        """
        if not self.done:
            self.run()
        if filename is None:
            filename = 'data/{}_tips_{}_ogs.pkl'.\
                       format(self.stip_mult, self.og_mult)
        else:
            filename = 'data/' + filename
        with open(filename, 'wb') as pklfile:
            pickle.dump(self.__dict__, pklfile)
        return

    def load(self, filename):
        """
        Re-creates an ensemble from a pkl file.

        Parameters
        ----------
        filename : str
            The location of the pkl file.
        """
        # with open(filename, 'rb') as pklfile:
        #     tmp = pickle.load(pklfile)
        # self.__dict__.update(tmp)
        self.__dict__.update(pickle.load(open(filename, 'rb')))
        return


class SimulationPair:
    """
    A pair of gap complexity simulations for the same system, one incorporating
    and the other disregarding an outer giant companion.

    Attributes
    ----------
    stip_mult : int
        The number of planets in the STIP.
    inc_scale : float
        The parameter of the Rayleigh distribution from which inclinations are
        sampled, in radians.
    og_inc : float
        The orbital inclination of the companion, in radians.
    simulation_time : int
        Number of minimum dynamical timescales for which the simulation is run.
    stellar_radius : float
        Radius of the star, in G = 1 and M = 1 length units.
    rng_seed : int
        Seed for random number generation.
    with_og : rebound.Simulation
        N-body simulation with the companion.
    wout_og : rebound.Simulation
        N-body simulation without the companion.
    min_dynamical_timescale : float
        The minimum dynamical timescale in the system (orbital period of the
        innermost planet).
    time : numpy.ndarray
        Range of time over which the Laplace--Lagrange solution is computed.
    og_mass : float
        Mass of the companion, in units of the primary mass.
    og_sma : float
        Semi-major axis of the companion, in G = 1 and M = 1 length units.
    sys_with : celmech.secular.LaplaceLagrangeSystem
        Secular dynamics simulation with the companion.
    sys_wout : celmech.secular.LaplaceLagrangeSystem
        Secular dynamics simulation without the companion.
    sol_with : dict
        Laplace--Lagrange solution with the companion.
    sol_wout : dict
        Laplace--Lagrange solution without the companion.
    mean_stip_inclination_with : float
        Inclination of the total STIP angular momentum in the current reference
        frame, for the system with the companion.
    mean_stip_inclination_wout : float
        Inclination of the total STIP angular momentum in the current reference
        frame, for the system without the companion.
    gc_with : numpy.ndarray
        Gap complexity with time for the system with the companion.
    gc_wout : numpy.ndarray
        Gap complexity with time for the system without the companion.
    mean_gc_with : float
        Time-averaged gap complexity for the system with the companion.
    mean_gc_wout : float
        Time-averaged gap complexity for the system without the companion.
    
    Methods
    -------
    add(m, a)
        Adds a companion, but only to one of the simulations.
    get_ll_systems()
        Retrieves the celmech Laplace--Lagrange systems.
    get_ll_solutions()
        Solves Lagrange's planetary equations.
    _run_nbody(simulation, binfile)
        Runs an N-body simulation using the whfast integrator in rebound, and
        returns the time series results.
    _nbody_inclination_solution(results)
        Produces the data for a plot of relative orbital inclinations with time
        from an N-body simulation.
    nbody_with(binfile='.archive.bin')
        Runs a whfast simulation of the system with an outer companion
        included.
    nbody_wout(binfile='.archive.bin')
        Runs a whfast simulation of the system without the inclusion of an
        outer companion.
    get_gap_complexity(simulation, solution)
        Gets the list of currently transiting STIP planets as a function of
        time, then calculates the corresponding gap complexity with respect to
        time.
    get_gap_complexities()
        Calculates the time-dependent gap complexity of the system based on the
        orbital inclinations.
    _get_nan_limits(time_series, time=None)
        Retrieves temporal ranges in which a time series (e.g., gap complexity)
        is a NaN.
    plot(orb_elem='inc', save=False)
        Makes a publication-quality plot of the inclination and gap complexity
        evolution obtained by solving the planetary equations.
    plot_nbody_and_secular(legend=False, save=False)
        Plots the N-body and Laplace--Lagrange solutions for the inclination
        evolution in the system, to evaluate the accuracy of the latter.
    """

    def __init__(self, stip_mult, inc_scale=2.5, og_inc=10.0,
                 simulation_time=1.0e9, stellar_radius=0.005, rng_seed=None):
        self.stip_mult = stip_mult
        self.inc_scale = inc_scale
        self.og_inc = og_inc
        self.simulation_time = simulation_time
        self.stellar_radius = stellar_radius
        self.rng_seed = rng_seed
        self.with_og = make_system(stip_mult, inc_scale, rng_seed)
        self.wout_og = self.with_og.copy()
        self.min_dynamical_timescale = self.wout_og.particles[1].P
        self.time = np.linspace(0.0,
                        self.simulation_time * self.min_dynamical_timescale,
                        10_000) / self.min_dynamical_timescale
        self._nbody_with_done = False
        self._nbody_wout_done = False
        return

    def _inclination_canonical_coords(self, inc_from_stip):
        canonical_heliocentric_inclinations = [
            o.inc for o in reb_orbits(self.with_og, 'canonical heliocentric')
        ]
        inc_from_stip *= to_radians
        inc_from_stip -= np.mean(canonical_heliocentric_inclinations)
        return inc_from_stip

    def add(self, m, a):
        """
        Adds a companion, but only to one of the simulations.

        Parameters
        ----------
        m : float
            Mass of the outer giant companion, in units of the primary mass.
        a : float
            Semi-major axis of the outer giant companion, in G = 1, M = 1
            units.
        """
        self.og_mass = m
        self.og_sma  = a
        # og_inc_from_stip = np.mean(
        #     [p.inc for p in self.with_og.particles[1:]]
        # ) + self.og_inc * to_radians
        # og_inc_from_stip = self._inclination_canonical_coords(self.og_inc)
        og_inc_from_stip = self.og_inc * to_radians
        try:
            self.with_og.add(
                m=m, a=a, e=0.0, inc=og_inc_from_stip, pomega=0.0, Omega=0.0
            )
            # self.with_og.move_to_hel()
        except AssertionError as err:
            print('The OG is within the STIP. Semi-major axes are:')
            print([p.a for p in self.with_og.particles[1:]])
        return

    def get_ll_systems(self):
        """
        Retrieves the celmech Laplace--Lagrange systems.

        Returns
        -------
        celmech.secular.LaplaceLagrangeSystem
            The Laplace--Lagrange system with an outer giant.
        celmech.secular.LaplaceLagrangeSystem
            The Laplace--Lagrange system without an outer giant.
        """
        self.sys_with = LaplaceLagrangeSystem.from_Simulation(self.with_og)
        self.sys_wout = LaplaceLagrangeSystem.from_Simulation(self.wout_og)
        return self.sys_with, self.sys_wout

    def get_ll_solutions(self):
        """
        Solves Lagrange's planetary equations.

        Returns
        -------
        dict
            The Laplace--Lagrange solution for the system with an outer giant.
        dict
            The Laplace--Lagrange solution for the system without an outer
            giant.
        """
        self.sol_with = self.sys_with.secular_solution(self.time)
        self.sol_wout = self.sys_wout.secular_solution(self.time)
        return self.sol_with, self.sol_wout
    
    def _run_nbody(self, simulation, binfile):
        """
        Runs an N-body simulation using the whfast integrator in rebound, and
        returns the time series results.

        Parameters
        ----------
        simulation : rebound.Simulation
            The simulation we want to integrate forward.
        binfile : str
            The name for the binary simulation archive file.
        
        Returns
        -------
        dict
            Archived rebound simulation data, read in with celmech.
        """
        self._nbody_with_done = False
        self._nbody_wout_done = False
        simulation.dt = 0.01 * self.min_dynamical_timescale
        simulation.integrator = 'whfast'
        simulation.max_exit_distance = 20.0
        simulation.ri_whfast.coordinates = 'democraticheliocentric'
        simulation.save_to_file(binfile, interval=1e2, delete_file=True)
        simulation.integrate(self.simulation_time, exact_finish_time=False)
        return get_simarchive_integration_results(binfile)
    
    def _nbody_inclination_solution(self, results):
        """
        Produces the data for a plot of relative orbital inclinations with time
        from an N-body simulation.

        Parameters
        ----------
        results : dict
            Archived rebound simulation data, read in with celmech.
        
        Returns
        -------
        tuple (numpy.ndarray) (float)
            Two arrays: one containing the simulation output times and the
            other containing the inclinations for each planet in the system
            at each output time.
        """
        relative_incs = 1.0 * results['inc']
        # for ii in range(len(relative_incs)):
        #     relative_incs[ii] -= np.mean(results['inc'][:self.stip_mult], axis=0)
        return results['time'], relative_incs
    
    def nbody_with(self, binfile='.archive.bin'):
        """
        Runs a whfast simulation of the system with an outer companion
        included.

        Returns
        -------
        tuple (numpy.ndarray) (float)
            Two arrays: one containing the simulation output times and the
            other containing the inclinations for each planet in the system
            at each output time.
        """
        results = None
        if self._nbody_with_done:
            results = get_simarchive_integration_results(binfile)
        else:
            results = self._run_nbody(self.with_og, binfile)
            self._nbody_with_done = True
        return self._nbody_inclination_solution(results)

    def nbody_wout(self, binfile='.archive.bin'):
        """
        Runs a whfast simulation of the system without the inclusion of an
        outer companion.

        Returns
        -------
        tuple (numpy.ndarray) (float)
            Two arrays: one containing the simulation output times and the
            other containing the inclinations for each planet in the system
            at each output time.
        """
        results = None
        if self._nbody_wout_done:
            results = get_simarchive_integration_results(binfile)
        else:
            results = self._run_nbody(self.wout_og, binfile)
            self._nbody_wout_done = True
        return self._nbody_inclination_solution(results)
    
    # def _run_nbody(self, simulation):
    #     """
    #     Runs an N-body simulation using the whfast integrator in rebound.

    #     Parameters
    #     ----------
    #     simulation : rebound.Simulation
    #         The simulation we want to integrate forward.
        
    #     Returns
    #     -------
    #     tuple (numpy.ndarray) (float)
    #         Two arrays: one containing the simulation output times and the
    #         other containing the inclinations for each planet in the system
    #         at each output time.
    #     """
    #     innermost_period = simulation.particles[1].P
    #     simulation.dt = 0.01 * innermost_period # 1% of innermost period
    #     end_time = 1.0e6 * innermost_period # 1 million orbits

    #     # number of whfast steps
    #     num_steps = (end_time + simulation.dt // 2) // simulation.dt
    #     num_steps = int(num_steps)

    #     num_outputs = 0
    #     nbody_time = np.empty((num_steps,))
    #     nbody_inclinations = np.empty((self.stip_mult, num_steps))

    #     def inclination_heartbeat(sim_pointer):
    #         """
    #         Heartbeat function for N-body simulations to check the accuracy of the
    #         Laplace--Lagrange solution.

    #         Parameters
    #         ----------
    #         sim_pointer : rebound.particle.LP_Simulation
    #             Pointer to the rebound simulation.
    #         """
    #         global num_outputs, nbody_time, nbody_inclinations
    #         sim = sim_pointer.contents
    #         # if num_outputs == 0:
    #         #     j = 0 # timestep index
    #         if sim.t > (num_outputs * sim.dt * 1.0e4): # one output every 1e4 orbits
    #             # nbody_time[j] = sim.t
    #             nbody_time[num_outputs] = sim.t
    #             for i in range(self.stip_mult):
    #                 mean_stip_inc = np.mean(
    #                     [p.inc for p in sim.particles[1:self.stip_mult]]
    #                 )
    #                 nbody_inclinations[i][num_outputs] = \
    #                 sim.particles[i+1].inc - mean_stip_inc
    #             num_outputs += 1
    #         return

    #     # inclination_heartbeat = cdll.LoadLibrary('inclination_heartbeat.so')
        
    #     simulation.heartbeat = inclination_heartbeat
    #     # simulation.integrate(simulation.t + 1)
    #     simulation.integrate(end_time, exact_finish_time=False)

    #     # nbody_time = c_double.in_dll(inclination_heartbeat, 'nbody_time')
    #     # nbody_inclinations = c_double.in_dll(inclination_heartbeat, 'nbody_inclinations')
    #     nbody_time /= innermost_period # convert to same units as secular time
    #     return nbody_time, nbody_inclinations

    def longest_secular_period(self):
        """
        Retrieves the period of the longest secular cycle in the
        Laplace--Lagrange system.

        Returns
        -------
        float
            The period of the longest secular cycle, in simulation units.
        """
        frequencies = np.abs(self.sys_with.inclination_eigenvalues())
        frequencies = np.sort(frequencies)
        # Use the second-smallest frequency; one will always be ~zero
        return 2.0 * np.pi / frequencies[1]
    
    def amd(self, primary, secondary):
        """
        Calculates the angular momentum deficit (AMD) of one body in the
        system.

        Parameters
        ----------
        primary : rebound.Particle
            The primary (star) in the system.
        secondary : rebound.Particle
            The body whose AMD we want to calculate with respect to the given
            primary.

        Returns
        -------
        float
            The AMD of the body in question. This quantity is dimensionless,
            since we use G = 1 natural units.
        """
        result = secondary.m * np.sqrt(primary.m * secondary.a) * \
                 (1.0 - np.sqrt(1.0 - secondary.e * secondary.e) * \
                 np.cos(secondary.inc))
        return result
    
    def total_amd(self):
        """
        Calculates the total angular momentum deficit (AMD) of the system, both
        including and excluding the outer companion.

        Returns
        -------
        tuple (float)
            The AMD of the STIP alone, and the AMD of the whole system, both
            dimensionless (G = 1 units).
        """
        amds = np.empty((self.stip_mult+1,))
        for ii in range(self.stip_mult+1):
            amds[ii] = self.amd(
                self.with_og.particles[0], self.with_og.particles[ii+1]
            )
        amd_with = np.sum(amds)
        amd_wout = np.sum(amds[:-1])
        return amd_with, amd_wout
    
    def amd_metric(self):
        """
        Computes a quantity related to the AMD that might correlate with gap
        complexity amplification.

        Returns
        -------
        float
            The value of this quantity.
        """
        return self.total_amd()[0] # Just the total AMD of the STIP

        # # AMD, weighted toward higher semi-major axis
        # per_planet = np.empty((self.stip_mult,))
        # for ii in range(self.stip_mult):
        #     per_planet[ii] = (ii + 1) * self.amd(
        #         self.with_og.particles[0], self.with_og.particles[ii+1]
        #     )
        # return np.sum(per_planet)

        # # Normalized AMD (Turrini+ 2020)
        # amd_per = np.empty((self.stip_mult,))
        # cam_per = np.emtpy((self.stip_mult,))
        # for ii in range(self.stip_mult):
        #     amd_per[ii] = self.amd(
        #         self.with_og.particles[0], self.with_og.particles[ii+1]
        #     )
        #     cam_per[ii] = self.with_og.particles[ii+1].m * \
        #                   np.sqrt(self.with_og.particles[ii+1].a)
        # return np.sum(amd_per) / np.sum(cam_per)

        # # Just the inclination of the outermost STIP planet
        # return self.with_og.particles[-2].inc * to_degrees

        # The variance in the STIP inclinations
        # return np.var([p.inc for p in self.wout_og.particles[1:]]) * to_degrees
    
    def get_gap_complexity(self, simulation, solution):
        """
        Gets the list of currently transiting STIP planets as a function of time,
        then calculates the corresponding gap complexity with respect to time.

        Parameters
        ----------
        simulation : rebound.Simulation
            The N-body simulation of the system.
        solution : dict
            The Laplace--Lagrange solution calculated by celmech.
        
        Returns
        -------
        numpy.ndarray
            The sequence of observed gap complexities with time.
        """
        obs_gc = np.array([])
        stip_inc = np.array([])
        max_incs = [
            max_transiting_inclination(self.stellar_radius, simulation.particles[j+1].a)
            for j in range(self.stip_mult)
        ]
        for tt in range(len(self.time)):
            incs = solution['inc'][0:self.stip_mult, tt]
            stip_mean_inc = np.mean(incs)
            incs -= stip_mean_inc # Rotate into mean inclination plane of STIP
            obs_idx = [j for j in range(self.stip_mult)
                       if np.abs(incs[j] * to_degrees) < max_incs[j]]
            obs_per = [simulation.particles[j+1].P for j in obs_idx]
            obs_gc  = np.append(obs_gc, gap_complexity(obs_per))
            stip_inc = np.append(stip_inc, stip_mean_inc)

        if simulation.N == self.stip_mult+1:
            self.mean_stip_inclination_wout = stip_inc
        else:
            self.mean_stip_inclination_with = stip_inc
        return obs_gc

    def get_gap_complexities(self):
        """
        Calculates the time-dependent gap complexity of the system based on the
        orbital inclinations.

        Returns
        -------
        numpy.ndarray
            Gap complexity of the system with an outer giant.
        numpy.ndarray
            Gap complexity of the system without an outer giant.
        """
        self.gc_with = self.get_gap_complexity(self.with_og, self.sol_with)
        self.gc_wout = self.get_gap_complexity(self.wout_og, self.sol_wout)
        try:
            self.mean_gc_with = np.nanmean(self.gc_with)
            self.mean_gc_wout = np.nanmean(self.gc_wout)
        except RuntimeWarning as warning:
            # print()
            # print(i, j)
            # print(warning)
            # print()
            pass
        # Replace NaNs with zeros to make plots without holes
        if np.isnan(self.mean_gc_with):
            self.mean_gc_with = 0.
        if np.isnan(self.mean_gc_wout):
            self.mean_gc_wout = 0.
        return self.gc_with, self.gc_wout

    def _get_nan_limits(self, time_series, time=None):
        """
        Retrieves temporal ranges in which a time series (e.g., gap complexity)
        is a NaN.

        Parameters
        ----------
        time_series : numpy.ndarray
            The time series in question.
        time : numpy.ndarray
            The corresponding range in time.
        
        Returns
        -------
        list (tuple) (float)
            Start and end times for each span of time in which the time series
            is a NaN.
        """
        if time is not None:
            pass
        else:
            time = self.time
        # Get periods of time in which a time series is NaN
        nan_idx = np.argwhere(np.isnan(time_series))
        start_idx = []
        end_idx = []
        nan_times = []
        for i, idx in enumerate(nan_idx):
            try:
                if i == 0:
                    start_idx.append(idx)
                elif i == len(nan_idx) - 1:
                    end_idx.append(idx)
                else:
                    if nan_idx[i-1] != idx - 1:
                        start_idx.append(idx)
                    if nan_idx[i+1] != idx + 1:
                        end_idx.append(idx)
            except IndexError as err:
                print(nan_idx.shape)
                print(err)
                pass
        idx_pairs = zip(start_idx, end_idx)
        for pair in idx_pairs:
            nan_times.append((time[pair[0]], time[pair[1]]))
        return nan_times

    def _discontinuities(self, series):
        """
        Finds the indices at which a time series has jump discontinuities, so
        we can prevent them from showing up in a plot.

        Parameters
        ----------
        time_series : numpy.ndarray
            The time series in question.
        time : numpy.ndarray
            The corresponding range in time.
        
        Returns
        -------
        numpy.ndarray (int)
            Temporal indices at which discontinuities occur.
        """
        discontinuous = np.abs(series[1:] - series[:-1]) > 0.01
        discontinuous = np.roll(discontinuous, 1)
        return np.argwhere(discontinuous).ravel()

    def _draw_discontinuous_curve(self, ax, time, time_series, color, lw):
        brs = self._discontinuities(time_series)
        if len(brs) == 0:
            ax.plot(time, time_series, c=color, lw=lw)
        else:
            time_slices = [
                time[brs[ii]+1:brs[ii+1]] for ii in range(len(brs)-1)
            ]
            tseries_slices = [
                time_series[brs[ii]+1:brs[ii+1]] for ii in range(len(brs)-1)
            ]
            for ii in range(len(time_slices)):
                ax.plot(time_slices[ii], tseries_slices[ii], c=color, lw=lw)
        return

    def plot(self, orb_elem='inc', save=False):
        """
        Makes a publication-quality plot of the inclination and gap complexity
        evolution obtained by solving the planetary equations.

        Parameters
        ----------
        orb_elem : str
            The orbital element whose evolution is plotted. Defaults to
            inclination; only change for a quick check on the other elements.
        save : bool
            Save the figure as a PDF?
        """
        # Plot params
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['font.family'] = 'cmu-serif'
        mpl.rcParams['font.size'] = 10
        palette = seaborn.color_palette('colorblind', self.with_og.N)
        alpha = 1.0
        lw = 2

        # Get periods of time in which gap complexity is a NaN
        plot_time = self.time
        # plot_time = self.time / 1.0e4
        nan_times_with = self._get_nan_limits(self.gc_with, plot_time)
        nan_times_wout = self._get_nan_limits(self.gc_wout, plot_time)

        # Calculation and plotting
        fig, axes = plt.subplots(2, 2, dpi=200, constrained_layout=True)
        for ii, particle in enumerate(self.wout_og.particles[1:]):
            max_inc = max_transiting_inclination(self.stellar_radius,
                                                 particle.a)
            evol_with = self.sol_with[orb_elem][ii, :]
            evol_wout = self.sol_wout[orb_elem][ii, :]
            axes[0, 0].plot(plot_time,
                            evol_wout * to_degrees, c=palette[ii],
                            alpha=alpha, lw=lw)
            axes[0, 1].plot(plot_time,
                            evol_with * to_degrees, c=palette[ii],
                            alpha=alpha, lw=lw, label=str(ii+1))
            if orb_elem == 'inc':
                for ax in axes[0, :]:
                    ax.axhline(+max_inc, c=palette[ii], ls='dotted')
                    ax.axhline(-max_inc, c=palette[ii], ls='dotted')
            # if max_inc > max_max_inc:
            #     max_max_inc = max_inc
        axes[1, 0].scatter(plot_time, self.gc_wout, c='k', s=0.25)
        axes[1, 1].scatter(plot_time, self.gc_with, c='k', s=0.25)
        # self._draw_discontinuous_curve(
        #     axes[1, 0], plot_time, self.gc_with, 'k', lw
        # )
        # self._draw_discontinuous_curve(
        #     axes[1, 1], plot_time, self.gc_wout, 'k', lw
        # )
        # [print(gc) for gc in self.gc_with]
        for pair in nan_times_wout:
            _min, _max = (float(lim) for lim in pair)
            axes[1, 0].axvspan(_min, _max, color='silver')
        for pair in nan_times_with:
            _min, _max = (float(lim) for lim in pair)
            axes[1, 1].axvspan(_min, _max, color='silver')
        # for ax in axes[:, 1]:
        #     ax.set_ylim(0.0, 1.0)
        axes[0, 0].set_ylabel(r'$I$ (deg)')
        axes[0, 1].set_ylabel(r'$I$ (deg)')
        axes[1, 0].set_ylabel(r'$\tilde{\mathcal{C}}$')
        axes[1, 1].set_ylabel(r'$\tilde{\mathcal{C}}$')
        for ax in axes.ravel():
            ax.set_xlim(plot_time.min(), plot_time.max())
        for ax in axes[0, :]:
            # ax.text(0.025, 0.9, 'With OG', transform=ax.transAxes)
            ax.xaxis.set_ticklabels([])
        for ax in axes[1, :]:
            ax.set_xlabel(r'$t/t_\mathrm{dyn}$')
            # ax.set_xlabel(r'$t/t_\mathrm{dyn} \times 10^4$')
            # ax.text(0.025, 0.9, 'Without OG', transform=ax.transAxes)
        # for ax in axes[:, 0]:
        #     ylim = max_max_inc + 2.0
        #     ax.set_ylim(-ylim, ylim)
        ymax = print_cmax(False)[self.stip_mult-1]
        for ax in axes[1, :]:
            ax.set_ylim(-0.02, ymax+0.15)
        axes[1, 0].text(0.56, 0.885,
                        r'$\langle \tilde{\mathcal{C}} \rangle_1 =$' +
                        ' {avg:.6f}'.format(avg=self.mean_gc_wout),
                        bbox=dict(facecolor='w', edgecolor='k'),
                        transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.56, 0.885,
                        r'$\langle \tilde{\mathcal{C}} \rangle_2 =$' +
                        ' {avg:.6f}'.format(avg=self.mean_gc_with),
                        bbox=dict(facecolor='w', edgecolor='k'),
                        transform=axes[1, 1].transAxes)
        # fig.tight_layout()
        if save:
            fig.savefig('new-plot.pdf')
        return
    
    def plot_nbody_and_secular(self, legend=False, save=False):
        """
        Plots the N-body and Laplace--Lagrange solutions for the inclination
        evolution in the system, to evaluate the accuracy of the latter.

        Parameters
        ----------
        legend : bool
            Make legend in the first subplot?
        save : bool
            Save the figure as a PDF?
        """
        nbody_time, nbody_inclinations = None, None

        # Plot params
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['font.family'] = 'cmu-serif'
        mpl.rcParams['font.size'] = 10
        palette = seaborn.color_palette('colorblind', self.with_og.N)
        alpha = 1.0
        lw = 2

        # Run N-body simulation
        try:
            nbody_time, nbody_inclinations = self.nbody_with()
        except rebound.Escape:
            print('UNSTABLE: Ejected planet.')
            return
        except rebound.Encounter:
            print('UNSTABLE: Close encounter or collision.')
            return

        # Plotting -- note this plot includes the OG for a full comparison of
        # the dynamics
        fig, axes = plt.subplots(2, 1, dpi=200, constrained_layout=True)
        for ii, particle in enumerate(self.with_og.particles[1:]):
            secular_inclination = self.sol_with['inc'][ii, :]
            axes[0].plot(self.time, secular_inclination * to_degrees,
                         c=palette[ii], alpha=alpha, lw=lw, label=str(ii+1))
            axes[1].plot(nbody_time, nbody_inclinations[ii] * to_degrees,
                         c=palette[ii], alpha=alpha, lw=lw)
        for ax in axes:
            ax.set_xlim(self.time.min(), self.time.max())
            ax.set_ylabel(r'$I$ (deg)')
        axes[0].set_title('Laplace--Lagrange solution')
        axes[1].set_title(r'$N$-body solution')
        axes[1].set_xlabel('Simulation time')
        if legend:
            axes[0].legend()
        if save:
            fig.savefig('nbody-secular-compare.pdf')
        return


def unspindle(y):
    """
    Given a real number y, calculates the argument x of y = x^x.
    """
    x = np.exp(lambertw(np.log(y)))
    x = np.asfarray(x)
    return x


def max_transiting_inclination(stellar_radius, sma):
    """
    Calculates the maximum transiting inclination of a planet in the system.

    Parameters
    ----------
    stellar_radius : float
        The radius of the star in au.
    sma : float
        The semi-major axis of the planet in au.
    
    Returns
    -------
    float
        The maximum orbital inclination the planet can achieve while
        transiting, in degrees.
    """
    max_inc_radians = np.arctan(stellar_radius / sma)
    max_inc_degrees = max_inc_radians * to_degrees
    return max_inc_degrees


def scaling_matrix(N):
    """
    Returns the scaling matrix, of which the semi-major axes are elements of an
    eigenvector.

    Parameters
    ----------
    N : int
        The multiplicity of the system.
    
    Returns
    -------
    numpy.ndarray (float)
        The scaling matrix.
    """
    matrix = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            if i == 0 or i == N - 1:
                if j == i:
                    matrix[i][j] = 1.0
                else:
                    matrix[i][j] = 0.0
            else:
                if i == j - 1 or i == j + 1:
                    matrix[i][j] = 0.5
                else:
                    matrix[i][j] = 0.0
    return matrix


def get_semi_major_axes(N, a_inner, a_outer):
    """
    Calculates the set of semi-major axes in a STIP, given an innermost and
    outermost semi-major axis, and following the orbital period ratio trend
    identified by Weiss+2018.

    Parameters
    ----------
    N : int
        The STIP multiplicity.
    a_inner : float
        The semi-major axis (simulation units) of the innermost planet.
    a_outer : float
        The semi-major axis (simulation units) of the outermost planet.
    
    Returns
    -------
    numpy.ndarray (float)
        The semi-major axis of every planet in the STIP.
    """
    smas = None
    eig = np.linalg.eig(scaling_matrix(N))
    eigvecs = eig.eigenvectors.T
    for ii, vec in enumerate(eigvecs):
        if eig.eigenvalues[ii] == 1.0 and np.all(vec[1:] < vec[:-1]):
            smas = vec
    if smas is None:
        raise NewError('No solution for semi-major axes!')
    # The final entry in the vector is always zero, so we can manipulate it
    # to have the proportionalities we want.
    scale = np.log(a_inner / a_outer) / smas[0]
    smas = np.exp(smas * scale) * a_outer
    return smas


def make_system(stip_mult, inc_scale, rng_seed):
    """
    Initializes a REBOUND simulation containing the STIP.

    Parameters
    ----------
    stip_mult : int
        The number of planets in the STIP.
    inc_scale : float
        Mean value in the Rayleigh distribution for mutual inclinations, in
        radians.
    rng_seed : int
        Seed for random number generation.
    
    Returns
    -------
    rebound.Simulation
        An N-body simulation containing the primary and the STIP.
    """
    sim = rebound.Simulation()
    rng = np.random.default_rng(rng_seed)
    smas = get_semi_major_axes(stip_mult, 0.1, 0.5)
    incs = [rng.rayleigh(inc_scale) * to_radians for _ in range(stip_mult)]
    incs -= np.mean(incs) # Ensure I = 0 corresponds to the mean STIP angular momentum plane
    # Note that everything is in G = 1 units
    sim.add(m=1.0)
    for ii in range(stip_mult):
        sim.add(
            m = 1.0e-5,
            a = smas[ii],
            e = 0.0,
            inc = incs[ii], # sample inclinations from Fabrycky+ 2014 dist
            # inc = 0.25 * to_radians, # constant inclinations
            pomega = 0.0,
            Omega  = 0.0,
        )
    sim.move_to_hel()
    return sim


def print_cmax(tabular=True):
    """
    Values of C_max calculated by Gilbert & Fabrycky (2020).

    Parameters
    ----------
    tabular : bool
        Return as a pandas dataframe?
    
    Returns
    -------
    pandas.DataFrame or pandas.Series
        Tabulated C_max values in the specified format.
    """
    data = {
        'n': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'C_max': [0.0, 0.0, 0.106, 0.212, 0.291, 0.350, 0.398, 0.437, 0.469, 0.497]
    }
    if tabular:
        return pd.DataFrame.from_dict(data)
    else:
        return data['C_max']


def gap_complexity(periods):
    """
    Calculates the gap complexity from a list of observed periods.

    Parameters
    ----------
    periods : list (float)
        List of orbital periods in the system.
    
    Returns
    -------
    float
        The derived gap complexity.
    """
    cmax = print_cmax(tabular=False)
    if len(periods) < 3:
        return np.nan
    else:
        max_period_ratio = periods[-1] / periods[0]
        p_star = []
        for ii, period in enumerate(periods[:-1]):
            period_ratio = periods[ii+1] / period
            p_star.append(np.log(period_ratio) / np.log(max_period_ratio))
        ngaps = len(p_star)
        entropy = sum([p * np.log(p) for p in p_star])
        disequilibrium = sum([(p - 1.0/ngaps) ** 2 for p in p_star])
        complexity = -1.0/cmax[ngaps] * entropy * disequilibrium
        return complexity


def mutual_hill_radius_particles(primary, planet1, planet2):
    """
    Mutual Hill radius of two adjacent planets, passed as rebound Particles.

    Parameters
    ----------
    primary : rebound.Particle
        The primary of the system.
    planet1 : rebound.Particle
        The inner planet of the pair.
    planet2 : rebound.Particle
        The outer planet of the pair.
    
    Returns
    -------
    float
        The mutual Hill radius, in the same untis as the semi-major axes of
        the planets.
    """
    return mutual_hill_radius(
        primary.m, planet1.m, planet2.m, planet1.a, planet2.a
    )


def mutual_hill_radius(mprimary, m1, m2, a1, a2):
    """
    Mutual Hill radius of two adjacent planets, from masses and semi-major
    axes.

    Parameters
    ----------
    mprimary : float
        The mass of the primary.
    m1 : float
        The mass of the inner planet.
    m2 : float
        The mass of the outer planet.
    a1 : float
        The semi-major axis of the inner planet.
    a2 : float
        The semi-major axis of the outer planet.
    
    Returns
    -------
    float
        The mutual Hill radius, in the same untis as the semi-major axes of
        the planets.
    """
    result = np.power((m1 + m2)/(3. * mprimary), 1./3.)
    result *= 0.5 * (a1 + a2)
    return result


def critical_mass_ratio(m1, a1, a2, num_hill_radii):
    """
    Calculates the maximum mass for the outer planet for which an adjacent pair
    of planets is "stable" (by our definition).

    Parameters
    ----------
    m1 : float
        The mass of the inner planet.
    a1 : float
        The semi-major axis of the inner planet.
    a2 : float
        The semi-major axis of the outer planet.
    num_hill_radii : float
        The minimum number of mutual Hill radii that must separate the two
        planets' semi-major axes.
    
    Returns
    -------
    float
        The maximum mass of the outer planet, in units of the primary mass.
    """
    result = 3.0 * (2.0 / num_hill_radii * (a2 - a1)/(a2 + a1)) ** 3 - m1
    return result


def minimum_difference(resonance_list, frequency_ratio):
    """
    Finds the minimum difference between a given frequency ratio and a list of
    resonances.

    Parameters
    ----------
    resonance_list : list (float)
        List of integer ratio values to test.
    frequency_ratio : float
        Ratio of two frequencies in the system.
    
    Returns
    -------
    float
        Minimum fractional difference between the given ratio and one of the
        integer ratios.
    """
    min_diff = 10.0
    for res in resonance_list:
        diff = np.abs(frequency_ratio - res) / res
        # diff = np.abs(frequency_ratio - res)
        if diff < min_diff:
            min_diff = diff
        else:
            pass
    return min_diff


def degree_of_commensurability(frequencies):
    """
    Takes a list of frequencies (e.g., inclination eigenfrequencies) and
    identifies the pair of most commensurate frequencies. Returns the deviation
    from an integer ratio.

    Parameters
    ----------
    frequencies : list (float)
        List of frequencies to test.
    
    Returns
    -------
    float
        Minimum fractional difference between an integer ratio and a ratio of
        given frequencies.
    """
    first_order_resonances = [
        1.0 / 2.0,
        2.0 / 3.0,
        3.0 / 4.0,
        4.0 / 5.0,
        5.0 / 6.0,
        6.0 / 7.0,
        7.0 / 8.0,
        8.0 / 9.0,
    ]
    second_order_resonances = [
        1.0 / 3.0,
        3.0 / 5.0,
        5.0 / 7.0,
        7.0 / 9.0,
    ]
    third_order_resonances = [
        1.0 / 4.0,
        2.0 / 5.0,
        4.0 / 7.0,
        5.0 / 8.0,
    ]
    fourth_order_resonances = [
        1.0 / 5.0,
        3.0 / 7.0,
        5.0 / 9.0,
    ]
    frequency_pairs = [
        (f1, f2) \
        for ii, f1 in enumerate(frequencies) \
        for f2 in frequencies[ii+1:]
    ]
    ratios = []
    for f1, f2 in frequency_pairs:
        ratio = f1 / f2
        min_diff_first_order  = minimum_difference(first_order_resonances,
                                                   ratio)
        min_diff_second_order = minimum_difference(second_order_resonances,
                                                   ratio)
        min_diff_third_order  = minimum_difference(third_order_resonances,
                                                   ratio)
        min_diff_fourth_order = minimum_difference(fourth_order_resonances,
                                                   ratio)
        min_diff_overall = min([
            min_diff_first_order,
            min_diff_second_order,
            min_diff_third_order,
            min_diff_fourth_order,
        ])
    return min_diff_overall


def load_ensemble_pair(filename):
    """
    Reconstructs a pickled EnsemblePair.

    Parameters
    ----------
    filename : str
        The address of the pkl file.
    
    Returns
    -------
    EnsemblePair
        The saved simulation ensemble.
    """
    new_pair = EnsemblePair()
    new_pair.load(filename)
    return new_pair
    