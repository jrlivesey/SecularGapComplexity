import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from gap_complexity_utils import EnsemblePair


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'cmu-serif'
mpl.rcParams['font.size'] = 16


vlim = None
rng_seed = 420


def plot_resonances(num, stip_mult, og_inc):
    global vlim
    ensemble = EnsemblePair(num, stip_mult, og_inc=og_inc, rng_seed=rng_seed,
                            vlim=vlim)
    ensemble.sample('grid')
    fig, _ = ensemble.secular_resonance_heatmap(output=True)
    fig.savefig('resonance_{}_tips_{}_oginc.pdf'.\
                format(str(stip_mult), str(round(og_inc))))
    if vlim is None:
        vlim = np.abs(max([ensemble.proximity_to_resonance.min(),
                           ensemble.proximity_to_resonance.max()]))
    else:
        pass
    return


def main(argv):
    num_per_ensemble = int(argv[1])
    plot_resonances(num_per_ensemble, 4, 40.0)
    plot_resonances(num_per_ensemble, 4, 20.0)
    plot_resonances(num_per_ensemble, 4, 10.0)
    plot_resonances(num_per_ensemble, 5, 10.0)
    plot_resonances(num_per_ensemble, 6, 10.0)
    return


if __name__ == '__main__':
    main(sys.argv)
