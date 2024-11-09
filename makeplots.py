import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
from copy import deepcopy
import sys
import os
import celmech
from gap_complexity_utils import EnsemblePair


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'cmu-serif'
mpl.rcParams['font.size'] = 16


sample_ensemble = None
sample_gc_with = None
sample_gc_wout = None
# vlim = None
vlim = 0.3


# Set arbitrary seed for numpy.random
rng_seed = 47


# def gc_mesh(arr, clabel, destname, cmap=mpl.cm.Spectral):
#     _a, _m = np.meshgrid(sample_ensemble.alpha_array,
#                          sample_ensemble.mass_array)
#     vlim = 0.15
#     fig, ax = plt.subplots(1, 1, dpi=200)
#     ax.pcolormesh(_a, _m, arr, vmin=-vlim, vmax=vlim, cmap=cmap)
#     cb = plt.colorbar(
#         plt.cm.ScalarMappable(
#             norm=mpl.colors.Normalize(-vlim, vlim),
#             cmap=cmap
#         ),
#         ax=ax
#     )
#     ax.set_xlabel(r'$\alpha^{-1}$')
#     ax.set_ylabel(r'$m_\mathrm{OG}/m_\star$')
#     cb.set_label(clabel)
#     ax.set_yscale('log')
#     fig.tight_layout()
#     fig.savefig(destname)
#     return


def run_grid(num, stip_mult, og_inc, sma_max=10., keep_sample=False):
    global sample_ensemble, sample_gc_with, sample_gc_wout
    global vlim
    new_seed = rng_seed * stip_mult * og_inc
    ensemble = EnsemblePair(num, stip_mult, og_inc=og_inc, rng_seed=new_seed,
                            vlim=vlim, save_simulation_pairs=False,
                            randomize_inclinations=True)
    ensemble.sample('grid',
        mass_min=1.0e-5,
        mass_max=0.5,
        sma_min=1.,
        sma_max=250.,
        ascale='log',
    )
    ensemble.run()
    fig, _ = ensemble.heatmaps(mask=True, draw_boundary=False, cmap=mpl.cm.BrBG, output=True)
    fig.savefig('{}_tips_{}_oginc.pdf'.\
        format(str(stip_mult), str(round(og_inc))))
    amd_plot(ensemble, stip_mult, og_inc)
    if keep_sample:
        sample_ensemble = deepcopy(ensemble)
        sample_gc_with, sample_gc_wout = ensemble.gc_with, ensemble.gc_wout
    else:
        pass
    # Whatever the min/max are become the cmap limits going forward
    if vlim is None:
        vlim = np.abs(max([ensemble.gc_err.min(), ensemble.gc_err.max()]))
        # vlim = np.mean(ensemble.gc_err) + 2 * np.std(ensemble.gc_err) # since distribution should be centered roughly on zero, this will show most of the variability
    else:
        pass
    return


def amd_plot(ensemble, stip_mult, og_inc):
    amd_metric = ensemble.amd_metric.ravel()
    gc_diff = np.abs(ensemble.gc_err.ravel())
    idx = np.argsort(gc_diff)
    amd_metric = amd_metric[idx]
    gc_diff = gc_diff[idx]

    fig, ax = plt.subplots(1, 1)
    ax.scatter(amd_metric, gc_diff, c='k')
    ax.set_xscale('log')
    ax.set_xlabel('STIP AMD')
    ax.set_ylabel(r'$|\langle \mathcal{C} \rangle_2 - \langle \mathcal{C} \rangle_1|$')
    fig.tight_layout()
    fig.savefig('amd_{}_tips_{}_oginc.pdf'.\
                format(str(stip_mult), str(round(og_inc))))
    return


def main(argv):
    if len(argv) < 4:
        print(f'ERROR: Expected 4 arguments, received {len(argv)}.')
        print(
            f'Usage: python {argv[0]} [simulations per grid] [number of STIP planets] [OG inclination (deg)]'
        )
        exit(1)

    num_per_ensemble  = int(argv[1])
    stip_multiplicity = int(argv[2])
    giant_inclination = int(argv[3])
    sma_max = 10.
    if len(argv) > 4:
        sma_max = int(argv[4])

    # gc_with = np.array([])
    # gc_wout = np.array([])

    run_grid(num_per_ensemble, stip_multiplicity, giant_inclination, sma_max)
    # run_grid(num_per_ensemble, 4, 40.0)
    # run_grid(num_per_ensemble, 4, 20.0)
    # run_grid(num_per_ensemble, 4, 10.0, keep_sample=True)    
    # run_grid(num_per_ensemble, 5, 10.0)
    # run_grid(num_per_ensemble, 6, 10.0)

    # gc_mesh(
    #     sample_gc_with,
    #     r'$\langle \tilde{\mathcal{C}} \rangle_2$',
    #     'gc_with.pdf'    
    # )
    # gc_mesh(
    #     sample_gc_wout,
    #     r'$\langle \tilde{\mathcal{C}} \rangle_1$',
    #     'gc_wout.pdf'
    # )
    # gc_mesh(
    #     sample_gc_with - sample_gc_wout,
    #     r'$\langle \tilde{\mathcal{C}} \rangle_2 - \langle \tilde{\mathcal{C}} \rangle_1$',
    #     'gc_diff.pdf'
    # )


if __name__ == '__main__':
    main(sys.argv)
