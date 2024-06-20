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
vlim = None


# Set arbitrary seed for numpy.random
rng_seed = 420


def gc_mesh(arr, clabel, destname, cmap=mpl.cm.Spectral):
    _a, _m = np.meshgrid(sample_ensemble.alpha_array,
                         sample_ensemble.mass_array)
    vlim = 0.15
    fig, ax = plt.subplots(1, 1, dpi=200)
    ax.pcolormesh(_a, _m, arr, vmin=-vlim, vmax=vlim, cmap=cmap)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(
            norm=mpl.colors.Normalize(-vlim, vlim),
            cmap=cmap
        ),
        ax=ax
    )
    ax.set_xlabel(r'$\alpha^{-1}$')
    ax.set_ylabel(r'$m_\mathrm{OG}/m_\star$')
    cb.set_label(clabel)
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(destname)
    return


def run_grid(num, stip_mult, og_inc, keep_sample=False):
    global sample_ensemble, sample_gc_with, sample_gc_wout
    global vlim
    ensemble = EnsemblePair(num, stip_mult, og_inc=og_inc, rng_seed=rng_seed,
                            vlim=vlim)
    ensemble.sample('grid')
    ensemble.run()
    fig, _ = ensemble.heatmaps(output=True)
    fig.savefig('{}_tips_{}_oginc.pdf'.\
        format(str(stip_mult), str(round(og_inc))))
    if keep_sample:
        sample_ensemble = deepcopy(ensemble)
        sample_gc_with, sample_gc_wout = ensemble.gc_with, ensemble.gc_wout
    else:
        pass
    # Whatever the min/max are become the cmap limits going forward
    if vlim is None:
        vlim = np.abs(max([ensemble.gc_err.min(), ensemble.gc_err.max()]))
    else:
        pass
    return


def main(argv):
    num_per_ensemble = int(argv[1])

    og_incs = [10.0, 20.0, 40.0]
    # n_stip  = [4, 5, 6]
    n_stip = [5, 6] # since we already have a 4-TIP case
    gc_with = np.array([])
    gc_wout = np.array([])
    # hms = []

    # processes = [
    #     # mp.Process(target=run_grid, args=(num_per_ensemble, 4, 10.0)),
    #     mp.Process(target=run_grid, args=(num_per_ensemble, 4, 20.0)),
    #     mp.Process(target=run_grid, args=(num_per_ensemble, 4, 40.0)),
    #     mp.Process(target=run_grid, args=(num_per_ensemble, 5, 10.0)),
    #     mp.Process(target=run_grid, args=(num_per_ensemble, 6, 10.0)),
    # ]
    # for p in processes: p.start()

    # # Keep this one independent so that it can modify global variables
    # run_grid(num_per_ensemble, 4, 10.0, True)
    # for p in processes: p.join()

    run_grid(num_per_ensemble, 4, 40.0)
    run_grid(num_per_ensemble, 4, 20.0)
    run_grid(num_per_ensemble, 4, 10.0, True)    
    run_grid(num_per_ensemble, 5, 10.0)
    run_grid(num_per_ensemble, 6, 10.0)

    # for ii, inc in enumerate(og_incs):
    #     csv_file = 'data/4_tips_{}_oginc.csv'.format(str(round(inc)))
    #     # if os.path.exists(csv_file):
    #     #     ensemble_data = pd.read_csv(csv_file)
    #     #     gc_with = []
    #     #     gc_wout = []
    #     # else:
    #     ensemble = EnsemblePair(num_per_ensemble, 4, og_inc=inc,
    #                             rng_seed=rng_seed)
    #     ensemble.sample('grid')
    #     ensemble.run()
    #     fig, ax = ensemble.heatmaps(output=True)
    #     fig.savefig('4_tips_{}_oginc.pdf'.format(str(round(inc))))
    #     # hms.append(hm)
    #     if ensemble.stip_mult == 4 and ensemble.og_inc == 10.0:
    #         sample_gc_with = ensemble.gc_with
    #         sample_gc_wout = ensemble.gc_wout
    #     gc_with = np.append(gc_with, ensemble.gc_with)
    #     gc_wout = np.append(gc_wout, ensemble.gc_wout)
    #     # gc_with = np.append(gc_with, np.array([np.nanmean(simpair.gc_with) for simpair in ensemble.pairs.ravel()]))
    #     # gc_wout = np.append(gc_wout, np.array([np.nanmean(simpair.gc_wout) for simpair in ensemble.pairs.ravel()]))
    #     df = ensemble.to_dataframe(True, csv_file)

    # for ii, n in enumerate(n_stip):
    #     csv_file = 'data/{}_tips_1_ogs.csv'.format(str(n))
    #     # if os.path.exists(csv_file):
    #     #     ensemble_data = pd.read_csv(csv_file)
    #     #     gc_with = []
    #     #     gc_wout = []
    #     # else:
    #     ensemble = EnsemblePair(num_per_ensemble, n, rng_seed=rng_seed)
    #     ensemble.sample('grid')
    #     ensemble.run()
    #     fig, ax = ensemble.heatmaps(output=True)
    #     fig.savefig('{}_tips_1_ogs.pdf'.format(str(n)))
    #     # hms.append(hm)
    #     gc_with = np.append(gc_with, ensemble.gc_with)
    #     gc_wout = np.append(gc_wout, ensemble.gc_wout)
    #     # gc_with = np.append(gc_with, np.array([np.nanmean(simpair.gc_with) for simpair in ensemble.pairs.ravel()]))
    #     # gc_wout = np.append(gc_wout, np.array([np.nanmean(simpair.gc_wout) for simpair in ensemble.pairs.ravel()]))
    #     df = ensemble.to_dataframe(True, csv_file)

    gc_mesh(
        sample_gc_with,
        r'$\langle \tilde{\mathcal{C}} \rangle_2$',
        'gc_with.pdf'    
    )
    gc_mesh(
        sample_gc_wout,
        r'$\langle \tilde{\mathcal{C}} \rangle_1$',
        'gc_wout.pdf'
    )
    gc_mesh(
        sample_gc_with - sample_gc_wout,
        r'$\langle \tilde{\mathcal{C}} \rangle_2 - \langle \tilde{\mathcal{C}} \rangle_1$',
        'gc_diff.pdf'
    )

if __name__ == '__main__':
    main(sys.argv)
