"""
Functions and data structures for finding the gap complexity of a secularly
evolving planetary system.
Joseph Livesey, 2024
"""


import numpy as np
import astropy.units as u
import astropy.constants as const
import rebound as reb
import pandas as pd
import itertools
import pickle
from copy import deepcopy

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn

import celmech
from celmech import Poincare, PoincareHamiltonian
from celmech.secular import LaplaceLagrangeSystem


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

    def __init__(self, num_simulations=100, stip_mult=4, og_mult=1,
                 inc_scale=2.5, og_inc=10.0, rng_seed=None, vlim=0.15,
                 save_simulation_pairs=True):
        # Class attributes
        self.num_simulations = num_simulations
        self.stip_mult = stip_mult
        self.og_mult   = og_mult
        self.inc_scale = inc_scale
        self.og_inc = og_inc
        if rng_seed is not None:
            self.rng_seed = rng_seed
            self.rng = np.random.default_rng(rng_seed)
        else:
            pass
        self.vlim = vlim
        self.save_simulation_pairs = save_simulation_pairs
        self.init = False
        self.done = False

        # Plot params
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['font.family'] = 'cmu-serif'
        mpl.rcParams['font.size'] = 16
        return

    def _rng_func(self, dist, *args):
        def rn():
            return dist(*args)
        return rn

    def set_og_mass_dist(self, dist, *args):
        self.og_mass_dist = self._rng_func(dist, *args)
        return

    def set_og_sma_dist(self, dist, *args):
        self.og_sma_dist = self._rng_func(dist, *args)
        return

    def _random_orbit_separation(self, sma_first, sma_max):
        stdev = 0.1
        # Set mean new semimajor axis at location of 2:1 MMR (idk)
        loc = (4.0**(1.0/3) - 1.0) * sma_first
        delta_a = self.rng.normal(loc, stdev)
        if delta_a <= 0.0:
            return self._random_orbit_separation(sma_first, sma_max)
        else:
            return delta_a

    def sample(self, method='random', mass_min=1.0e-3, mass_max=0.5,
               sma_min=0.75, sma_max=10.0):
        self.method = method
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
                                          simulation_time=1.0e6,
                                          rng_seed=self.rng_seed)
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
            self.sma_array  = np.linspace(sma_min, sma_max, side_len)
            for i in range(side_len):
                for j in range(side_len):
                    sim_pair = SimulationPair(self.stip_mult, self.inc_scale,
                                              self.og_inc, stellar_radius=0.005,
                                              rng_seed=self.rng_seed)
                    sim_pair.add(m=self.mass_array[i], a=self.sma_array[j])
                    self.pairs[i][j] = sim_pair # need 2D for grid sampling
        else:
            raise NewError('Sampling method must be `grid` or `random`.')
        self.init = True
        return

    def run(self):
        if not self.init:
            self.sample()
        if self.method == 'random':
            for sim_pair in self.pairs:
                sim_pair.get_ll_systems()
                sim_pair.get_ll_solutions()
                sim_pair.get_gap_complexities()
            self.gc_with = np.empty((self.num_simulations))
            self.gc_wout = np.empty((self.num_simulations))
            for i in range(self.num_simulations):
                self.gc_with[i] = np.nanmean(self.pairs[i].gc_with)
                self.gc_wout[i] = np.nanmean(self.pairs[i].gc_wout)
                if not self.save_simulation_pairs:
                    self.pairs[i] = None
        elif self.method == 'grid':
            self.gc_with = np.empty(self.pairs.shape)
            self.gc_wout = np.empty(self.pairs.shape)
            for i, j in itertools.product(
                range(int(np.sqrt(self.num_simulations))),
                range(int(np.sqrt(self.num_simulations)))
            ):
                self.pairs[i][j].get_ll_systems()
                self.pairs[i][j].get_ll_solutions()
                self.pairs[i][j].get_gap_complexities()
                self.gc_with[i][j] = np.nanmean(self.pairs[i][j].gc_with)
                self.gc_wout[i][j] = np.nanmean(self.pairs[i][j].gc_wout)
                if not self.save_simulation_pairs:
                    self.pairs[i][j] = None
        else:
            raise NewError('Sampling method must be `grid` or `random`.')
        self.done = True
        return
    
    def _get_proximity_to_resonance(self):
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
                self.proximity_to_resonance[i][j] = degree_of_commensurability(
                    self.pairs[i][j].sys_with.inclination_eigenvalues()
                )
        return

    def histogram(self, save=False):
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

    def heatmaps(self, save=False, output=False, key1=None, key2=None):
        cmap = mpl.cm.Spectral
        if self.method != 'grid':
            raise NewError('Heatmaps can be made only for data sampled over a grid.')
        if not self.done:
            self.run()
        else:
            pass
        # fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=200, sharex=True)
        fig, ax = plt.subplots(1, 1, dpi=200)
        self.alpha_array = self.sma_array / (10.0 ** (2.0/3 - 1.0)) # ratio between OG sma and that of outermost TIP
        self.gc_err = self.gc_with - self.gc_wout
        if self.vlim is None:
            self.vlim = np.abs(max([self.gc_err.min(), self.gc_err.max()]))
        else:
            pass
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
            ax=ax
        )
        ax.set_yscale('log')
        # for ax in axes:
        #     ax.set_xlabel('OG mass')
        # axes[0].set_ylabel(r'$\alpha$')
        ax.set_xlabel(r'$\alpha^{-1}$')
        ax.set_ylabel(r'$m_\mathrm{OG}/m_\star$')
        # ax.invert_yaxis()
        cb.set_label(
            r'$\langle \tilde{\mathcal{C}} \rangle_2 - \langle \tilde{\mathcal{C}} \rangle_1 $')
        fig.tight_layout()
        if save:
            fig.savefig('new-heatmaps.pdf')
        if output:
            return fig, ax
        else:
            return
    
    def secular_resonance_heatmap(self, save=False, output=False):
        self._get_proximity_to_resonance()
        cmap = mpl.cm.magma
        fig, ax = plt.subplots(1, 1, dpi=200)
        self.alpha_array = self.sma_array / (10.0 ** (2.0/3 - 1.0)) # ratio between OG sma and that of outermost TIP
        if self.vlim is None:
            self.vlim = np.abs(max([self.proximity_to_resonance.min(),
                                    self.proximity_to_resonance.max()]))
        else:
            pass
        _a, _m = np.meshgrid(self.alpha_array, self.mass_array)
        ax.pcolormesh(_a, _m, self.proximity_to_resonance, vmin=-self.vlim,
                      vmax=self.vlim, cmap=cmap)
        cb = plt.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap,
                                  norm=mpl.colors.Normalize(-self.vlim,
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
        # with open(filename, 'rb') as pklfile:
        #     tmp = pickle.load(pklfile)
        # self.__dict__.update(tmp)
        self.__dict__.update(pickle.load(open(filename, 'rb')))
        return


class SimulationPair:

    def __init__(self, stip_mult, inc_scale=2.5, og_inc=10.0,
                 simulation_time=50_000, stellar_radius=0.005, rng_seed=None):
        self.stip_mult = stip_mult
        self.inc_scale = inc_scale
        self.og_inc = og_inc
        self.simulation_time = simulation_time
        self.stellar_radius = stellar_radius
        self.rng_seed = rng_seed
        self.with_og = make_system(stip_mult, inc_scale, rng_seed)
        self.wout_og = deepcopy(self.with_og)
        self.min_dynamical_timescale = self.wout_og.particles[1].P
        self.time = np.linspace(0.0,
                        self.simulation_time * self.min_dynamical_timescale,
                        10_000) / self.min_dynamical_timescale
        return

    def add(self, m, a):
        """
        Adds a companion, but only to the sim without giants.
        """
        self.og_mass = m
        self.og_sma  = a
        # self.og_inc = rng.rayleigh(self.inc_scale * to_radians)
        self.with_og.add(m=m, a=a, inc=self.og_inc*to_radians)
        self.with_og.move_to_com()
        return

    def get_ll_systems(self):
        """
        Retrieves the celmech Laplace--Lagrange systems.
        """
        self.sys_with = LaplaceLagrangeSystem.from_Simulation(self.with_og)
        self.sys_wout = LaplaceLagrangeSystem.from_Simulation(self.wout_og)
        return self.sys_with, self.sys_wout

    def get_ll_solutions(self):
        """
        Solves Lagrange's planetary equations.
        """
        self.sol_with = self.sys_with.secular_solution(self.time)
        self.sol_wout = self.sys_wout.secular_solution(self.time)
        return self.sol_with, self.sol_wout
    
    def get_gap_complexity(self, simulation, solution):
        """
        Gets the list of currently transiting STIP planets as a function of time,
        then calculates the corresponding gap complexity with respect to time.
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
        self.gc_with = self.get_gap_complexity(self.with_og, self.sol_with)
        self.gc_wout = self.get_gap_complexity(self.wout_og, self.sol_wout)
        try:
            self.mean_gc_with = np.nanmean(self.gc_with)
            self.mean_gc_wout = np.nanmean(self.gc_wout)
        except RuntimeWarning as warning:
            print()
            print(i, j)
            print(warning)
            print()
        return self.gc_with, self.gc_wout

    def _get_nan_limits(self, time_series, time=None):
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
                if nan_idx[i-1] != idx - 1:
                    start_idx.append(idx)
                if nan_idx[i+1] != idx + 1:
                    end_idx.append(idx)
            except IndexError:
                pass
        idx_pairs = zip(start_idx, end_idx)
        for pair in idx_pairs:
            nan_times.append((time[pair[0]], time[pair[1]]))
        return nan_times

    def plot(self, orb_elem='inc', save=False):
        """
        Makes a publication-quality plot of the inclination and gap complexity
        evolution obtained by solving the planetary equations.
        """
        # Plot params
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['font.family'] = 'cmu-serif'
        mpl.rcParams['font.size'] = 16
        palette = seaborn.color_palette('colorblind', self.with_og.N)
        alpha = 1.0
        lw = 2

        # Get periods of time in which gap complexity is a NaN
        plot_time = self.time / 1.0e4
        nan_times_with = self._get_nan_limits(self.gc_with, plot_time)
        nan_times_wout = self._get_nan_limits(self.gc_wout, plot_time)

        # Calculation and plotting
        fig, axes = plt.subplots(2, 2, dpi=200, constrained_layout=True)
        # max_max_inc = 0.0
        for ii, particle in enumerate(self.wout_og.particles[1:]):
            max_inc = max_transiting_inclination(self.stellar_radius,
                                                 particle.a)
            evol_with = self.sol_with[orb_elem][ii, :]
            evol_wout = self.sol_wout[orb_elem][ii, :]
            axes[0, 0].plot(plot_time,
                            evol_with * to_degrees, c=palette[ii],
                            alpha=alpha, lw=lw, label=str(ii+1))
            axes[1, 0].plot(plot_time,
                            evol_wout * to_degrees, c=palette[ii],
                            alpha=alpha, lw=lw)
            if orb_elem == 'inc':
                for ax in axes[:, 0]:
                    ax.axhline(+max_inc, c=palette[ii], ls='dotted')
                    ax.axhline(-max_inc, c=palette[ii], ls='dotted')
            # if max_inc > max_max_inc:
            #     max_max_inc = max_inc
        axes[0, 1].plot(plot_time, self.gc_with, lw=lw)
        axes[1, 1].plot(plot_time, self.gc_wout, lw=lw)
        for pair in nan_times_with:
            _min, _max = (float(lim) for lim in pair)
            axes[0, 1].axvspan(_min, _max, color='silver')
        for pair in nan_times_wout:
            _min, _max = (float(lim) for lim in pair)
            axes[1, 1].axvspan(_min, _max, color='silver')
        # for ax in axes[:, 1]:
        #     ax.set_ylim(0.0, 1.0)
        axes[0, 0].set_ylabel(r'$I$ (deg)')
        axes[1, 0].set_ylabel(r'$I$ (deg)')
        axes[0, 1].set_ylabel(r'$\tilde{\mathcal{C}}$')
        axes[1, 1].set_ylabel(r'$\tilde{\mathcal{C}}$')
        for ax in axes.ravel():
            ax.set_xlim(plot_time.min(), plot_time.max())
        for ax in axes[0, :]:
            # ax.text(0.025, 0.9, 'With OG', transform=ax.transAxes)
            ax.xaxis.set_ticklabels([])
        for ax in axes[1, :]:
            ax.set_xlabel(r'$t/t_\mathrm{dyn} \times 10^{-4}$')
            # ax.text(0.025, 0.9, 'Without OG', transform=ax.transAxes)
        # for ax in axes[:, 0]:
        #     ylim = max_max_inc + 2.0
        #     ax.set_ylim(-ylim, ylim)
        ymax = print_cmax(False)[self.stip_mult-1]
        for ax in axes[:, 1]:
            ax.set_ylim(0.0, ymax+0.1)
        axes[0, 1].text(0.575, 0.9,
                        r'$\langle \tilde{\mathcal{C}} \rangle_2 =$' +
                        ' {avg:.6f}'.format(avg=self.mean_gc_with),
                        bbox=dict(facecolor='w', edgecolor='k'),
                        transform=axes[0, 1].transAxes)
        axes[1, 1].text(0.575, 0.9,
                        r'$\langle \tilde{\mathcal{C}} \rangle_1 =$' +
                        ' {avg:.6f}'.format(avg=self.mean_gc_wout),
                        bbox=dict(facecolor='w', edgecolor='k'),
                        transform=axes[1, 1].transAxes)
        # fig.tight_layout()
        if save:
            fig.savefig('new-plot.pdf')
        return
        

def max_transiting_inclination(stellar_radius, sma):
    """
    Calculates the maximum transiting inclination of a planet in the system.
    """
    max_inc_radians = np.arctan(stellar_radius / sma)
    max_inc_degrees = max_inc_radians * to_degrees
    return max_inc_degrees


def make_system(stip_mult, inc_scale, rng_seed):
    """
    Initializes a REBOUND simulation containing the STIP.
    """
    sim = reb.Simulation()
    rng = np.random.default_rng(rng_seed)
    # Note that everything is in G = 1 units
    P0 = 2.0 * np.pi
    sim.add(m=1.0)
    for ii in range(stip_mult):
        sim.add(
            m = 1.0e-5,
            # P = 0.01 * P0 * 10.0 ** ii, # Initially zero gap complexity in the STIP
            a = 0.1 * 10.0 ** (2./3. * (ii + 1)/stip_mult), # Initially zero gap complexity in the STIP
            e = 0.0,
            inc = rng.rayleigh(inc_scale) * to_radians, # sample inclinations from Fabrycky+ 2014 dist
            # inc = 0.25 * to_radians, # constant inclinations
            pomega = 0.0,
            Omega  = 0.0,
        )
    sim.move_to_com()
    return sim


def print_cmax(tabular=True):
    """
    Values of C_max calculated by Gilbert & Fabrycky (2020)
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
    """
    cmax = print_cmax(tabular=False)
    if len(periods) < 3:
        return np.nan
    else:
        max_period_ratio = periods[-1] / periods[0]
        p_star = []
        for ii, period in enumerate(periods[:-1]):
            period_ratio = periods[ii+1] / period
            p_star.append(np.log10(period_ratio) / np.log10(max_period_ratio))
        ngaps = len(p_star)
        entropy = sum([p * np.log10(p) for p in p_star])
        disequilibrium = sum([(p - 1.0/ngaps) ** 2 for p in p_star])
        complexity = -1.0/cmax[ngaps] * entropy * disequilibrium
        return complexity


def minimum_difference(resonance_list, frequency_ratio):
    """
    Finds the minimum difference between a given frequency ratio and a list of
    resonances.
    """
    min_diff = 10.0
    for res in resonance_list:
        diff = np.abs(frequency_ratio - res) / res
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
    Reconstructs ensemble pair from pkl file.
    """
    new_pair = EnsemblePair()
    new_pair.load(filename)
    return new_pair
    